"""
BotCore — Loop utama bot trading. AI-driven, bukan Python-driven.

Alur:
1. Ambil data mentah (Python)
2. Cek harga vs watchlist (Python)
3. Kalau ada trigger → panggil AI yang sesuai
4. AI buat keputusan → set watchlist baru atau eksekusi order
5. Python eksekusi order sesuai keputusan AI

State machine:
  h1_scan         → Hiura analisis H1 setiap siklus
  fvg_wait        → Tunggu FVG H1 disentuh (Python cek harga)
  idm_hunt        → Senanan cari IDM M5, set watchlist
  bos_guard       → Shina tunggu BOS/MSS M5 (Python cek harga)
  entry_sniper    → Yusuf tentukan entry, SL, TP
"""

import asyncio
import os
import json
import logging
from datetime import datetime, timezone
from groq import Groq

from data_provider import DataProvider
from ai_analysts import (
    hiura_h1_analysis, senanan_idm_hunt,
    shina_bos_mss, yusuf_entry, loss_debrief,
    katyusha_review, katyusha_post_trade
)
from watchlist_engine import WatchlistEngine
from rules_engine import RulesEngine
from logic_engine import LogicEngine
from memory_system import MemorySystem
from risk_manager import RiskManager
from trade_executor import TradeExecutor
import api_server

logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[logging.StreamHandler(), logging.FileHandler("logs/bot.log")],
)


class BotCore:
    def __init__(self):
        symbol      = os.getenv("TRADING_SYMBOL", "BTCUSDT")
        paper       = os.getenv("PAPER_TRADING", "true").lower() == "true"
        scan_sec    = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))

        # API keys
        groq_key  = (os.environ.get("GROQ_API_KEY") or
                     os.environ.get("GROQ_API_KEY_AI1") or
                     os.environ.get("GROQ_API_KEY_AI2") or
                     os.environ.get("GROQ_API_KEY_AI3"))
        if not groq_key:
            raise EnvironmentError("Tidak ada GROQ_API_KEY ditemukan!")

        key_ai1 = os.environ.get("GROQ_API_KEY_AI1") or groq_key
        key_ai2 = os.environ.get("GROQ_API_KEY_AI2") or groq_key
        key_ai3 = os.environ.get("GROQ_API_KEY_AI3") or groq_key

        # Models
        self.model_main = os.getenv("GROQ_MODEL_MAIN", "llama-3.3-70b-versatile")
        self.model_ai1  = os.getenv("GROQ_MODEL_AI1",  "llama-3.3-70b-versatile")
        self.model_ai2  = os.getenv("GROQ_MODEL_AI2",  "llama-3.3-70b-versatile")
        self.model_ai3  = os.getenv("GROQ_MODEL_AI3",  "llama-3.3-70b-versatile")
        self.model_json = os.getenv("GROQ_MODEL_JSON",  self.model_ai1)

        # Clients
        self.client_main = Groq(api_key=groq_key)
        self.clients = [
            Groq(api_key=key_ai1),
            Groq(api_key=key_ai2),
            Groq(api_key=key_ai3),
            Groq(api_key=groq_key),  # Yusuf pakai main
        ]
        self.models = [self.model_ai1, self.model_ai2, self.model_ai3, self.model_json]

        # Components
        self.data     = DataProvider(symbol)
        self.watchlist = WatchlistEngine()
        self.rules    = RulesEngine()
        self.logic    = LogicEngine()
        self.memory   = MemorySystem()
        self.risk     = RiskManager()
        self.executor = TradeExecutor(paper_mode=paper)

        self.scan_interval = scan_sec
        self.symbol = symbol
        self.paper  = paper
        self._katyusha_key      = os.environ.get("OPENROUTER_API_KEY", "")
        self._last_katyusha_ts  = 0.0   # timestamp review terakhir
        self._katyusha_interval = 5 * 3600  # 5 jam dalam detik

        # State
        self._phase         = "h1_scan"
        self._session_id    = ""
        self._prev_price    = 0.0
        self._last_bos_lvl  = 0.0  # hindari sesi duplikat BOS sama

        # Data dari tiap AI (disimpan antar fase)
        self._hiura_data  : dict = {}
        self._senanan_data: dict = {}
        self._shina_data  : dict = {}

        logger.info(f"BotCore ready | {symbol} | paper={paper}")
        logger.info(f"Models: main={self.model_main} | ai1={self.model_ai1} | "
                    f"ai2={self.model_ai2} | ai3={self.model_ai3}")

    # ── Helpers ─────────────────────────────────────────

    def _new_session(self, raw_data: dict, loss_ctx: str = "") -> str:
        sid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        self._session_id = sid
        api_server.start_session(sid, {"symbol": self.symbol, "price": raw_data.get("price")}, loss_ctx)
        return sid

    def _push(self, ai_key: str, name: str, msg: str, ronde: int = 1):
        if msg and self._session_id:
            api_server.push_message(ai_key, name, msg, ronde, self._session_id)

    def _finish_session(self, conclusion: dict):
        if self._session_id:
            api_server.finish_session(conclusion)

    def _reset(self, reason: str = ""):
        logger.info(f"[BOT] Reset ke h1_scan" + (f" — {reason}" if reason else ""))
        self._phase         = "h1_scan"
        self._hiura_data    = {}
        self._senanan_data  = {}
        self._shina_data    = {}
        self.watchlist.clear_untriggered()
        if self._session_id:
            self._finish_session({"consensus": reason or "reset"})
        self._session_id = ""

    def _logic_ctx(self) -> str:
        return self.logic.get_context_for_ai()

    # ── Phase Handlers ───────────────────────────────────

    def _katyusha_apply_changes(self, k_result: dict):
        """Apply semua perubahan rules dan logic dari Katyusha."""
        rules = self.rules.rules
        logic = self.logic.rules
        rules_dirty = False
        logic_dirty = False

        # RULES: changes (ubah nilai)
        for ch in k_result.get("rules_changes", []):
            sec, key, val = ch.get("section",""), ch.get("key",""), ch.get("new")
            if sec and key and val is not None:
                if sec not in rules:
                    rules[sec] = {}
                old_val = rules[sec].get(key, "N/A")
                rules[sec][key] = val
                rules_dirty = True
                logger.info(f"[KATYUSHA] rules CHANGE: {sec}.{key}: {old_val} → {val} | {ch.get('reason','')[:60]}")

        # RULES: adds (tambah key baru)
        for ch in k_result.get("rules_adds", []):
            sec, key, val = ch.get("section",""), ch.get("key",""), ch.get("value")
            if sec and key and val is not None:
                if sec not in rules:
                    rules[sec] = {}
                rules[sec][key] = val
                rules_dirty = True
                logger.info(f"[KATYUSHA] rules ADD: {sec}.{key} = {val} | {ch.get('reason','')[:60]}")

        # RULES: removes (hapus key)
        for ch in k_result.get("rules_removes", []):
            sec, key = ch.get("section",""), ch.get("key","")
            if sec and key and sec in rules and key in rules[sec]:
                del rules[sec][key]
                rules_dirty = True
                logger.info(f"[KATYUSHA] rules REMOVE: {sec}.{key} | {ch.get('reason','')[:60]}")

        if rules_dirty:
            rules["_update_reason"] = f"Katyusha review: {k_result.get('reasoning','')[:80]}"
            rules["_version"] = rules.get("_version", 1) + 1
            self.rules._save(rules)
            logger.info(f"[KATYUSHA] rules.json saved v{rules['_version']}")

        # LOGIC: changes
        for ch in k_result.get("logic_changes", []):
            sec, key, val = ch.get("section",""), ch.get("key",""), ch.get("new")
            if sec and key and val is not None and sec in logic:
                if isinstance(logic[sec], dict):
                    old_val = logic[sec].get(key, "N/A")
                    logic[sec][key] = val
                    logic_dirty = True
                    logger.info(f"[KATYUSHA] logic CHANGE: {sec}.{key}: {old_val} → {val} | {ch.get('reason','')[:60]}")

        # LOGIC: adds
        for ch in k_result.get("logic_adds", []):
            sec, key, val = ch.get("section",""), ch.get("key",""), ch.get("value")
            if sec and key and val is not None:
                if sec not in logic:
                    logic[sec] = {}
                logic[sec][key] = val
                logic_dirty = True
                logger.info(f"[KATYUSHA] logic ADD: {sec}.{key} = {val} | {ch.get('reason','')[:60]}")

        # LOGIC: removes
        for ch in k_result.get("logic_removes", []):
            sec, key = ch.get("section",""), ch.get("key","")
            if sec and key and sec in logic and isinstance(logic.get(sec), dict) and key in logic[sec]:
                del logic[sec][key]
                logic_dirty = True
                logger.info(f"[KATYUSHA] logic REMOVE: {sec}.{key} | {ch.get('reason','')[:60]}")

        if logic_dirty:
            logic["_update_reason"] = f"Katyusha review: {k_result.get('reasoning','')[:80]}"
            logic["_version"] = logic.get("_version", 1) + 1
            self.logic._save(logic)
            logger.info(f"[KATYUSHA] logic_rules.json saved v{logic['_version']}")

        total = (len(k_result.get("rules_changes",[])) + len(k_result.get("rules_adds",[])) +
                 len(k_result.get("rules_removes",[])) + len(k_result.get("logic_changes",[])) +
                 len(k_result.get("logic_adds",[])) + len(k_result.get("logic_removes",[])))
        if total > 0:
            logger.info(f"[KATYUSHA] Total {total} perubahan diterapkan")

    def _run_h1_scan(self, raw_data: dict):
        """Hiura analisis H1 setiap siklus. Buat sesi baru hanya saat BOS baru."""
        self.rules.reload()
        self.logic.reload()

        result = hiura_h1_analysis(
            self.clients[0], self.model_ai1,
            raw_data, self._logic_ctx()
        )
        self._hiura_data = result

        bos_level = result.get("bos_level", 0)
        bos_found = result.get("bos_found", False)
        msg       = result.get("chat_msg", "")

        if bos_found and bos_level != self._last_bos_lvl:
            # BOS baru → sesi baru
            self._last_bos_lvl = bos_level
            self._new_session(raw_data)
            self._push("ai1", "Hiura", msg, 1)

            # Pasang watchlist dari Hiura
            wl = result.get("watchlist", [])
            if wl:
                self.watchlist.clear_untriggered()
                for item in wl:
                    self.watchlist.add(
                        level=item["level"],
                        condition=item.get("condition", "touch"),
                        reason=item.get("reason", ""),
                        phase="fvg_wait",
                        session_ref=self._session_id,
                    )
                self._phase = "fvg_wait"
                logger.info(f"[HIURA] BOS {result.get('bos_type')} @ {bos_level:.2f} | "
                           f"{len(wl)} watchlist | Fase → fvg_wait")
            else:
                logger.info(f"[HIURA] BOS {bos_level:.2f} tapi tidak ada FVG fresh")

        elif bos_found:
            # BOS sama — push update ke sesi yang ada
            if msg and self._session_id:
                self._push("ai1", "Hiura", msg, 1)
            logger.info(f"[HIURA] BOS sama @ {bos_level:.2f} — update sesi")

        else:
            # Tidak ada BOS — live feed
            if msg:
                api_server.push_live_msg("ai1", "Hiura", msg)
            logger.info(f"[HIURA] Belum ada BOS | harga {raw_data.get('price')}")

    def _run_fvg_wait(self, raw_data: dict, triggered_items: list):
        """
        Python deteksi FVG disentuh berdasarkan harga vs watchlist.
        Saat trigger → Senanan mulai cari IDM M5.
        Juga pantau CHOCH H1.
        """
        price = raw_data.get("price", 0)

        # Cek CHOCH: harga close H1 melewati SL/SH referensi
        h1 = raw_data.get("h1", [])
        if h1:
            last = h1[-1]
            bos_type = self._hiura_data.get("bos_type", "")
            sl_ref = self._hiura_data.get("sl_before_bos", 0)
            sh_ref = self._hiura_data.get("sh_since_bos", 0)

            choch = False
            if bos_type == "bullish_bos" and sl_ref > 0 and last["c"] < sl_ref:
                choch = True
                msg = f"Hiura: CHOCH bearish — close {last['c']:.2f} di bawah SL ref {sl_ref:.2f}. Reset."
            elif bos_type == "bearish_bos" and sh_ref > 0 and last["c"] > sh_ref:
                choch = True
                msg = f"Hiura: CHOCH bullish — close {last['c']:.2f} di atas SH ref {sh_ref:.2f}. Reset."

            if choch:
                self._push("ai1", "Hiura", msg, 1)
                self._reset("choch")
                self._last_bos_lvl = 0.0
                return

        if not triggered_items:
            logger.info(f"[FVG WAIT] Harga {price:.2f} — tunggu FVG touch")
            return

        # Ada watchlist yang disentuh → FVG kena
        item = triggered_items[-1]
        logger.info(f"[FVG WAIT] FVG disentuh @ {item['level']:.2f} — panggil Senanan")

        sh = self._hiura_data.get("sh_since_bos", 0)
        sl = self._hiura_data.get("sl_before_bos", 0)
        bias = self._hiura_data.get("bias", "neutral")
        m5_dir = "bearish" if bias == "bullish" else "bullish"

        result = senanan_idm_hunt(
            self.clients[1], self.model_ai2,
            raw_data, sh, sl, m5_dir, bias,
            self._logic_ctx()
        )
        self._senanan_data = result
        self._push("ai2", "Senanan", result.get("chat_msg", ""), 2)

        wl = result.get("watchlist", [])
        if wl and result.get("idm_found"):
            self.watchlist.clear_untriggered()
            for item in wl:
                self.watchlist.add(
                    level=item["level"],
                    condition=item.get("condition", "touch"),
                    reason=item.get("reason", ""),
                    phase="bos_guard",
                    session_ref=self._session_id,
                )
            self._phase = "bos_guard"
            logger.info(f"[SENANAN] IDM ditemukan @ {result.get('watch_level',0):.2f} | Fase → bos_guard")
        else:
            logger.info("[SENANAN] IDM belum ditemukan — tetap di fvg_wait")

    def _run_bos_guard(self, raw_data: dict, triggered_items: list):
        """
        Tunggu IDM M5 disentuh, lalu Shina analisis BOS/MSS.
        """
        if not triggered_items:
            logger.info(f"[BOS GUARD] Tunggu IDM M5 touch | harga {raw_data.get('price'):.2f}")
            return

        item = triggered_items[-1]
        logger.info(f"[BOS GUARD] IDM M5 disentuh @ {item['level']:.2f} — panggil Shina")

        bias = self._hiura_data.get("bias", "neutral")
        sh   = self._hiura_data.get("sh_since_bos", 0)
        sl   = self._hiura_data.get("sl_before_bos", 0)

        result = shina_bos_mss(
            self.clients[2], self.model_ai3,
            raw_data, self._senanan_data,
            bias, sh, sl
        )
        self._shina_data = result
        self._push("ai3", "Shina", result.get("chat_msg", ""), 3)

        decision = result.get("decision", "wait")

        if decision == "entry":
            self._phase = "entry_sniper"
            logger.info("[SHINA] MSS terkonfirmasi → entry sniper")

        elif decision == "reset_idm":
            # MSS batal, cari IDM baru
            self._senanan_data = {}
            self._shina_data = {}
            self._phase = "fvg_wait"
            self.watchlist.clear_untriggered()
            logger.info("[SHINA] Reset IDM — kembali ke fvg_wait")

        else:
            # Wait — pasang watchlist freeze high/low
            wl = result.get("watchlist", [])
            if wl:
                self.watchlist.clear_untriggered()
                for w in wl:
                    self.watchlist.add(
                        level=w["level"],
                        condition=w.get("condition", "touch"),
                        reason=w.get("reason", ""),
                        phase="bos_guard",
                        session_ref=self._session_id,
                    )
            logger.info(f"[SHINA] Wait — watchlist freeze {len(wl)} level")

    def _run_entry_sniper(self, raw_data: dict):
        """Yusuf tentukan entry, SL, TP."""
        trade_mem = self.memory.get_recent_trades(limit=5)

        result = yusuf_entry(
            self.clients[3], self.model_json,
            raw_data, self._hiura_data, self._shina_data,
            trade_mem, self._logic_ctx()
        )
        self._push("ai4", "Yusuf", result.get("chat_msg", ""), 4)

        decision = result.get("decision", "skip")

        if decision == "entry":
            entry = result.get("entry", 0)
            sl    = result.get("sl", 0)
            tp    = result.get("tp", 0)
            conf  = result.get("confidence", 0)
            min_conf = self.rules.entry_min_confidence

            if entry > 0 and sl > 0 and tp > 0 and conf >= min_conf:
                direction = result.get("direction", "buy")

                # Sync balance
                live_bal = self.executor.get_account_balance()
                if live_bal > 0:
                    self.risk.account_balance = live_bal

                qty = self.risk.calculate_lot_size(entry=entry, stop_loss=sl, symbol=self.symbol)

                trade_res = self.executor.execute(
                    direction=direction, entry_price=entry,
                    stop_loss=sl, take_profit=tp, lot_size=qty,
                    signal={"entry_signal": direction, "confidence": conf,
                            "bias_reason": result.get("setup_type","")},
                )
                self.memory.log_trade(
                    symbol=self.symbol, direction=direction,
                    setup=result.get("setup_type",""), entry=entry,
                    sl=sl, tp=tp, rr=result.get("rr",0),
                    confidence=conf, notes=result.get("chat_msg",""),
                    trade_id=trade_res.get("trade_id"),
                )
                self._finish_session({
                    "consensus": "setuju_lanjut",
                    "entry_ideal_zona": str(entry),
                    "entry_ideal_timing": result.get("setup_type",""),
                    "risiko_utama": f"SL @ {sl:.2f}",
                    "avg_panel_confidence": conf,
                })
                logger.info(f"[YUSUF] ENTRY {direction.upper()} @ {entry:.2f} SL={sl:.2f} TP={tp:.2f}")
            else:
                logger.info(f"[YUSUF] Skip — conf={conf:.0%} < min={min_conf:.0%} atau level 0")
                result["decision"] = "skip"

        if decision == "skip":
            skip_reason = result.get("skip_reason", "MSS di luar FVG")
            self._push("ai4", "Yusuf", f"Yusuf: Skip — {skip_reason}. Kembali cari IDM.", 4)
            self._finish_session({"consensus": "skip_disarankan"})
            logger.info(f"[YUSUF] Skip: {skip_reason}")

        # Setelah entry atau skip → reset ke h1_scan
        self._reset("cycle_done")

    # ── Monitor Trade ────────────────────────────────────

    def _monitor_trades(self, raw_data: dict):
        """Cek trade yang sedang berjalan. Kalau ada yang close, proses debrief."""
        closed = self.executor.get_closed_trades()
        for trade in closed:
            result = trade.get("result")
            if result not in ("win", "loss"):
                continue

            self.memory.update_trade_result(trade)
            logger.info(f"[MONITOR] Trade closed: {result} PnL={trade.get('pnl',0):.4f}")

            if result == "loss":
                # Debrief + update rules + update logic
                all_ai_data = {
                    "hiura": self._hiura_data,
                    "senanan": self._senanan_data,
                    "shina": self._shina_data,
                }
                try:
                    debrief = loss_debrief(
                        self.clients,
                        self.models,
                        trade, all_ai_data
                    )
                    self.memory.log_error(
                        error=debrief.get("root_cause",""),
                        lesson=debrief.get("lesson",""),
                        context={"culprit": debrief.get("culprit"), "new_rule": debrief.get("new_rule")},
                    )
                    # Push debrief ke grup chat
                    sid = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_debrief"
                    api_server.start_session(sid, trade, "LOSS DEBRIEF")
                    for msg in debrief.get("chat_log", []):
                        api_server.push_message(
                            msg.get("ai","system"), msg.get("nama",""),
                            msg.get("pesan",""), msg.get("ronde",1), sid
                        )
                    api_server.finish_session({
                        "consensus": "loss_debrief",
                        "poin_debat": debrief.get("culprit",""),
                        "risiko_utama": debrief.get("root_cause",""),
                        "kondisi_reentry": debrief.get("new_rule",""),
                    })

                    # Update rules
                    self.rules.ai_update_on_loss(
                        self.client_main, self.model_json, trade, debrief
                    )
                    self.logic.ai_update_on_loss(
                        self.client_main, self.model_json, trade, debrief
                    )

                    # Katyusha post-trade evaluasi (lebih dalam dari Groq)
                    if self._katyusha_key:
                        k_post = katyusha_post_trade(
                            self._katyusha_key, trade,
                            {"hiura": self._hiura_data,
                             "senanan": self._senanan_data,
                             "shina": self._shina_data},
                            debrief,
                            self.rules.rules, self.logic.rules,
                        )
                        # Push ke chat
                        if k_post.get("chat_msg"):
                            sid_k = self._session_id or datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_katyusha"
                            if not self._session_id:
                                api_server.start_session(sid_k, trade, "KATYUSHA EVALUATION")
                                self._session_id = sid_k
                            self._push("katyusha", "Katyusha", k_post.get("chat_msg",""), 5)

                        # Apply rules changes dari Katyusha langsung
                        for change in k_post.get("rules_changes", []):
                            section = change.get("section","")
                            key     = change.get("key","")
                            val     = change.get("new")
                            if section and key and val is not None and section in self.rules.rules:
                                self.rules.rules[section][key] = val
                                logger.info(f"[KATYUSHA] Rules override: {section}.{key} = {val}")
                        if k_post.get("rules_changes"):
                            self.rules.rules["_update_reason"] = f"Katyusha override: {k_post.get('summary','')[:80]}"
                            self.rules.rules["_version"] = self.rules.rules.get("_version",1) + 1
                            self.rules._save(self.rules.rules)

                        # Apply logic changes dari Katyusha
                        for change in k_post.get("logic_changes", []):
                            section = change.get("section","")
                            key     = change.get("key","")
                            val     = change.get("new")
                            if section and key and val is not None and section in self.logic.rules:
                                if isinstance(self.logic.rules[section], dict):
                                    self.logic.rules[section][key] = val
                                    logger.info(f"[KATYUSHA] Logic override: {section}.{key} = {val}")
                        if k_post.get("logic_changes"):
                            self.logic.rules["_update_reason"] = f"Katyusha override: {k_post.get('summary','')[:80]}"
                            self.logic.rules["_version"] = self.logic.rules.get("_version",1) + 1
                            self.logic._save(self.logic.rules)
                except Exception as e:
                    logger.error(f"[DEBRIEF ERROR] {e}")

    # ── Main Loop ────────────────────────────────────────

    async def run(self):
        logger.info("=" * 60)
        logger.info(f"BotCore starting | {self.symbol} | {'PAPER' if self.paper else 'LIVE'}")
        logger.info("=" * 60)

        while True:
            try:
                # 1. Ambil data mentah
                raw = self.data.get_raw()
                if not raw:
                    logger.error("[BOT] Gagal ambil data market")
                    await asyncio.sleep(self.scan_interval)
                    continue

                price = raw["price"]

                # 2. Cek watchlist (Python cek harga, tidak ada analisis)
                triggered = self.watchlist.check(price, self._prev_price)
                if triggered:
                    for t in triggered:
                        logger.info(f"[TRIGGER] {t['condition'].upper()} @ {t['level']:.2f} | {t['reason'][:60]}")

                # 3. Katyusha review setiap 5 jam
                import time as _time
                now_ts = _time.time()
                if (self._katyusha_key and
                        now_ts - self._last_katyusha_ts >= self._katyusha_interval and
                        self._phase != "h1_scan"):
                    self._last_katyusha_ts = now_ts
                    logger.info("[KATYUSHA] Waktunya review 5 jam...")
                    bot_state = {
                        "phase":     self._phase,
                        "watchlist": self.watchlist.to_api_dict(),
                    }
                    k_result = katyusha_review(
                        self._katyusha_key, bot_state, raw,
                        {"hiura": self._hiura_data,
                         "senanan": self._senanan_data,
                         "shina": self._shina_data},
                        rules_current=self.rules.rules,
                        logic_current=self.logic.rules,
                    )
                    msg = k_result.get("chat_msg", "")
                    if msg:
                        self._push("katyusha", "Katyusha", msg, 5)

                    # Apply rules/logic edits dari Katyusha
                    self._katyusha_apply_changes(k_result)

                    # Override bot state
                    action = k_result.get("override_action", "none")
                    if action == "reset":
                        logger.info(f"[KATYUSHA] OVERRIDE: reset | {k_result.get('reasoning','')}")
                        self._last_bos_lvl = 0.0
                        self._reset("katyusha_override_reset")
                    elif action == "force_phase":
                        new_phase = k_result.get("override_phase", "")
                        if new_phase:
                            logger.info(f"[KATYUSHA] OVERRIDE: force_phase → {new_phase}")
                            self._phase = new_phase
                    elif action == "skip_entry":
                        logger.info("[KATYUSHA] OVERRIDE: skip_entry")
                        self._reset("katyusha_skip_entry")
                    elif k_result.get("verdict") == "warning":
                        logger.warning(f"[KATYUSHA] Warning: {k_result.get('reasoning','')}") 

                # 4. Dispatch ke fase yang tepat
                if self._phase == "h1_scan":
                    self._run_h1_scan(raw)

                elif self._phase == "fvg_wait":
                    self._run_fvg_wait(raw, triggered)

                elif self._phase == "bos_guard":
                    self._run_bos_guard(raw, triggered)

                elif self._phase == "entry_sniper":
                    self._run_entry_sniper(raw)

                # 4. Monitor trade aktif
                self._monitor_trades(raw)

                # 5. Update watchlist ke API
                api_server.update_watchlist(self.watchlist.to_api_dict())

                self._prev_price = price

                # Stats
                stats = self.memory.get_stats()
                logger.info(
                    f"[BOT] {price:.2f} | Fase: {self._phase} | "
                    f"Watchlist: {len(self.watchlist.get_active())} level | "
                    f"Trades: {stats['total_trades']} | WR: {stats['win_rate']:.0%}"
                )

            except KeyboardInterrupt:
                logger.info("Bot dihentikan")
                break
            except Exception as e:
                logger.error(f"[BOT] Error: {e}", exc_info=True)

            await asyncio.sleep(self.scan_interval)
