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
from prompt_engine import PromptEngine
from candle_replay import ReplayEngine, format_replay_for_ai
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
    def __init__(self, symbol: str = "", stagger_delay: float = 0.0):
        # Symbol bisa di-pass langsung (multi-symbol mode) atau via env (single mode)
        if not symbol:
            symbol = os.getenv("TRADING_SYMBOL", "BTCUSDT").strip()
        self.symbols = [symbol]  # untuk kompatibilitas
        self._stagger_delay = stagger_delay  # detik delay sebelum siklus pertama
        paper       = os.getenv("PAPER_TRADING", "true").lower() == "true"
        scan_sec    = int(os.getenv("SCAN_INTERVAL_SECONDS", "0"))  # 0 = auto align ke M5 close

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
        self.prompts  = PromptEngine()
        self.memory   = MemorySystem()
        self.risk     = RiskManager()
        self.executor = TradeExecutor(paper_mode=paper)

        self.scan_interval = scan_sec
        self.symbol = symbol
        self.paper  = paper
        self._replay = ReplayEngine(sw_left=8, sw_right=8, min_gap_pct=0.05)
        self._katyusha_key      = os.environ.get("OPENROUTER_API_KEY", "")
        self._last_katyusha_ts  = __import__("time").time()  # mulai dari sekarang, bukan epoch
        self._katyusha_interval = 1 * 3600  # 1 jam dalam detik

        # State
        self._phase         = "h1_scan"
        self._session_id    = ""
        self._prev_price    = 0.0
        self._last_bos_lvl  = 0.0  # hindari sesi duplikat BOS sama

        # Data dari tiap AI (disimpan antar fase)
        self._hiura_data  : dict = {}
        self._senanan_data: dict = {}
        self._shina_data  : dict = {}

        logger.info(f"BotCore ready | {symbol} | paper={paper} | symbols={self.symbols}")
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
            api_server.push_message(ai_key, name, msg, ronde, self._session_id, self.symbol)

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

    async def _wait_next_m5_close(self):
        """
        Tidur sampai tepat setelah candle M5 Bybit berikutnya close.
        Candle M5 Bybit close di menit: 0,5,10,15,...,55 setiap jam.
        Tambah 2 detik buffer agar candle sudah pasti terdaftar di API.
        Kalau SCAN_INTERVAL_SECONDS di-set manual, pakai itu saja.
        """
        if self.scan_interval > 0:
            await asyncio.sleep(self.scan_interval)
            return

        now = datetime.now(timezone.utc)
        seconds_into_interval = (now.minute % 5) * 60 + now.second
        seconds_to_close = (5 * 60) - seconds_into_interval
        wait = seconds_to_close + 2  # +2s buffer Bybit update klines

        logger.info(f"[{self.symbol}] Tunggu M5 close: {wait:.0f}s")
        await asyncio.sleep(wait)

    def _logic_ctx(self) -> str:
        return self.logic.get_context_for_ai()

    def _full_ctx(self, replay_text: str = "") -> dict:
        """Return semua JSON config + replay state untuk AI."""
        import json
        return {
            # Full JSON — tidak dipotong — semua AI harus lihat seluruh file
            "rules":      json.dumps({k:v for k,v in self.rules.rules.items()   if not k.startswith("_")}, ensure_ascii=False, separators=(',',':')),
            "logic":      json.dumps({k:v for k,v in self.logic.rules.items()   if not k.startswith("_")}, ensure_ascii=False, separators=(',',':')),
            "logic_raw":  {k:v for k,v in self.logic.rules.items() if not k.startswith("_")},
            "prompts":    json.dumps({k:v for k,v in self.prompts.prompts.items() if not k.startswith("_")}, ensure_ascii=False, separators=(',',':')),
            "replay_text": replay_text,  # hasil replay engine untuk Hiura
        }

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
            rules["_update_reason"] = f"Katyusha review: {k_result.get('reasoning','')}"
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
            logic["_update_reason"] = f"Katyusha review: {k_result.get('reasoning','')}"
            logic["_version"] = logic.get("_version", 1) + 1
            self.logic._save(logic)
            logger.info(f"[KATYUSHA] logic_rules.json saved v{logic['_version']}")

        # PROMPTS: update instruksi AI
        prompts = self.prompts.prompts.copy()
        prompts_dirty = False
        for ch in k_result.get("prompt_updates", []):
            ai_name = ch.get("ai", "").lower()
            field   = ch.get("field", "")
            value   = ch.get("value", "")
            if ai_name and field and value is not None and ai_name in prompts:
                old_val = prompts[ai_name].get(field, "")
                prompts[ai_name][field] = value
                prompts_dirty = True
                logger.info(f"[KATYUSHA] prompt UPDATE: {ai_name}.{field}: '{old_val[:40]}' → '{str(value)[:40]}'")

        if prompts_dirty:
            prompts["_update_reason"] = f"Katyusha review: {k_result.get('reasoning','')}"
            prompts["_version"] = prompts.get("_version", 1) + 1
            self.prompts.save(prompts)
            logger.info(f"[KATYUSHA] prompts.json saved v{prompts['_version']}")

        total = (len(k_result.get("rules_changes",[])) + len(k_result.get("rules_adds",[])) +
                 len(k_result.get("rules_removes",[])) + len(k_result.get("logic_changes",[])) +
                 len(k_result.get("logic_adds",[])) + len(k_result.get("logic_removes",[])) +
                 len(k_result.get("prompt_updates",[])))
        if total > 0:
            logger.info(f"[KATYUSHA] Total {total} perubahan diterapkan")

    def _run_h1_scan(self, raw_data: dict):
        """Hiura analisis H1 setiap siklus. Buat sesi baru hanya saat BOS baru."""
        self.rules.reload()
        self.logic.reload()
        # Sync replay engine params dari logic_rules.json
        _bc = self.logic.rules.get("find_bos_h1", {})
        self._replay.sw_left     = _bc.get("swing_left",  8)
        self._replay.sw_right    = _bc.get("swing_right", 8)
        _fc = self.logic.rules.get("find_fvg_h1", {})
        self._replay.min_gap_pct = _fc.get("min_gap_pct", 0.05)

        # Feed candle H1 ke ReplayEngine — replay kiri ke kanan
        h1_candles = raw_data.get("h1", [])
        replay_event = self._replay.replay_h1(h1_candles, current_price=raw_data.get('price', 0))
        event = replay_event.get("event", "none")

        # Format untuk Hiura
        replay_text = format_replay_for_ai(replay_event, self._replay)

        # Hiura hanya dipanggil Groq saat ada BOS baru (hemat token)
        # Status biasa → live feed langsung
        if event == "bos":
            result = hiura_h1_analysis(
                self.clients[0], self.model_ai1,
                raw_data, self._full_ctx(replay_text=replay_text),
                prompt_ctx=self.prompts.build_context('hiura')
            )
            self._hiura_data = result
            # Override dengan data dari replay (lebih akurat)
            bos_data = replay_event["data"].get("bos", {})
            self._hiura_data["bos_found"]     = True
            self._hiura_data["bos_type"]      = bos_data.get("type","")
            self._hiura_data["bos_level"]     = bos_data.get("swing_level", 0)
            self._hiura_data["bias"]          = bos_data.get("direction","")
            self._hiura_data["sh_since_bos"]  = bos_data.get("sh_before_bos") or bos_data.get("sh_since_bos",0)
            self._hiura_data["sl_before_bos"] = bos_data.get("sl_before_bos") or bos_data.get("sl_since_bos",0)
            self._hiura_data["fvg_list"]      = replay_event["data"].get("fvgs", [])
            msg = result.get("chat_msg","") or f"Hiura: {replay_text.split(chr(10))[0]}"
        elif event == "choch":
            msg = f"Hiura: CHOCH — {replay_event['data'].get('reason','')}. Reset scan."
            self._hiura_data = {"bos_found": False}
        elif event == "all_fvg_filled":
            msg = "Hiura: semua FVG sudah filled — reset, cari BOS baru."
            self._hiura_data = {"bos_found": False}
        else:
            # Status biasa — tidak panggil Groq
            state = self._replay.state
            if state == "SCAN_BOS":
                msg = f"Hiura: scan BOS H1... {raw_data.get('price',0):.4f}"
            elif state == "WAIT_FVG":
                fvg_cnt = len(self._replay.fvgs)
                msg = f"Hiura: BOS aktif, tunggu FVG touch | {fvg_cnt} FVG | {raw_data.get('price',0):.4f}"
            else:
                msg = f"Hiura: {state} | {raw_data.get('price',0):.4f}"

        bos_found = event == "bos"
        bos_level = self._hiura_data.get("bos_level", 0)

        if not msg:
            msg = f"Hiura: memantau {self.symbol} @ {raw_data.get('price',0):.4f}"
        api_server.push_live_msg("ai1", "Hiura", msg, self.symbol)

        if bos_found and abs(bos_level - self._last_bos_lvl) > 1e-9:  # toleransi floating point
            self._last_bos_lvl = bos_level
            bos_type = result.get("bos_type", "")
            sh       = result.get("sh_since_bos", 0) or result.get("bos_level", 0)
            sl       = result.get("sl_before_bos", 0)
            fvgs     = result.get("fvg_list", [])

            # Buat sesi baru + push Hiura
            self._new_session(raw_data)
            if msg:
                self._push("ai1", "Hiura", msg, 1)

            self.watchlist.clear_untriggered()

            # ── Watchlist 1: FVG touch → panggil AI-2 (IDM hunt) ──
            for fvg in [f for f in fvgs if not f.get("filled")][:3]:
                lvl   = fvg.get("mid") or (fvg["high"] + fvg["low"]) / 2
                self.watchlist.add(
                    level=lvl,
                    condition="touch",
                    reason=f"FVG {fvg['type']} {fvg['low']:.4f}–{fvg['high']:.4f} — sentuh = mulai IDM hunt",
                    phase="fvg_wait",
                    session_ref=self._session_id,
                    symbol=self.symbol,
                )

            # ── Watchlist 2: BOS invalidation → Hiura scan ulang (tanpa AI) ──
            if sl > 0:
                inv_cond = "break_below" if "bullish" in bos_type else "break_above"
                self.watchlist.add(
                    level=sl,
                    condition=inv_cond,
                    reason=f"BOS invalidation: {inv_cond} {sl:.4f} = CHOCH, scan ulang",
                    phase="bos_invalid",
                    session_ref=self._session_id,
                    symbol=self.symbol,
                )

            # ── Watchlist 3: Breakout jauh (beyond SH/SL) → Hiura scan BOS baru ──
            if sh > 0:
                ext_cond = "break_above" if "bullish" in bos_type else "break_below"
                self.watchlist.add(
                    level=sh,
                    condition=ext_cond,
                    reason=f"Breakout beyond SH/SL {sh:.4f} = BOS baru mungkin terbentuk",
                    phase="bos_breakout",
                    session_ref=self._session_id,
                    symbol=self.symbol,
                )

            self._phase = "fvg_wait"
            logger.info(f"[HIURA] BOS {bos_type} @ {bos_level:.4f} | "
                       f"FVG={len([f for f in fvgs if not f.get('filled')])} | "
                       f"Watchlist={len(self.watchlist.get_active())} level | Fase → fvg_wait")

        elif bos_found:
            if msg and self._session_id:
                self._push("ai1", "Hiura", msg, 1)
            logger.info(f"[HIURA] BOS sama @ {bos_level:.4f}")
        else:
            logger.info(f"[HIURA] Belum ada BOS | harga {raw_data.get('price')}")

    def _run_fvg_wait(self, raw_data: dict, triggered_items: list):
        """
        Tunggu trigger dari salah satu 3 watchlist:
          1. fvg_wait     → FVG disentuh → panggil AI-2
          2. bos_invalid  → BOS invalid (CHOCH) → reset tanpa AI
          3. bos_breakout → Breakout jauh → reset, Hiura scan ulang
        Python cek kondisi tambahan tanpa AI setiap siklus.
        """
        price = raw_data.get("price", 0)

        # Replay engine sudah handle FVG touch/filled/CHOCH di replay_h1()
        # Di sini kita tinggal cek event dari replay yang baru saja dijalankan
        # Kalau tidak ada trigger dari watchlist (level harga), cek replay state
        if not triggered_items:
            state = self._replay.state
            fvg_cnt = len(self._replay.fvgs)
            logger.info(f"[FVG WAIT] {price:.4f} | replay_state={state} | FVG={fvg_cnt}")
            return

        # Dispatch berdasarkan phase dari item yang trigger
        for item in triggered_items:
            phase_item = item.get("phase", "fvg_wait")
            lvl        = item.get("level", 0)

            # ── CHOCH / BOS Invalidation (tanpa AI) ──
            if phase_item == "bos_invalid":
                bos_type = self._hiura_data.get("bos_type", "")
                direction = "bearish" if "bullish" in bos_type else "bullish"
                msg = (f"Hiura: BOS invalid — harga {price:.4f} melewati level {lvl:.4f}. "
                       f"CHOCH {direction}. Reset, cari BOS baru.")
                api_server.push_live_msg("ai1", "Hiura", msg, self.symbol)
                if self._session_id:
                    self._push("ai1", "Hiura", msg, 1)
                logger.info(f"[FVG WAIT] BOS invalid @ {lvl:.4f} → reset")
                self._reset("bos_invalid")
                self._last_bos_lvl = 0.0
                return

            # ── Breakout jauh — mungkin ada BOS baru (tanpa AI dulu) ──
            elif phase_item == "bos_breakout":
                msg = (f"Hiura: breakout melewati {lvl:.4f}. "
                       f"BOS lama sudah jauh terlewat — scan ulang struktur H1.")
                api_server.push_live_msg("ai1", "Hiura", msg, self.symbol)
                if self._session_id:
                    self._push("ai1", "Hiura", msg, 1)
                logger.info(f"[FVG WAIT] Breakout @ {lvl:.4f} → reset ke h1_scan")
                self._reset("bos_breakout")
                self._last_bos_lvl = 0.0
                return

        # ── FVG disentuh → panggil Senanan (AI-2) ──
        fvg_items = [t for t in triggered_items if t.get("phase", "fvg_wait") == "fvg_wait"]
        if not fvg_items:
            return

        item = fvg_items[-1]
        logger.info(f"[FVG WAIT] FVG disentuh @ {item['level']:.4f} — panggil Senanan")

        if not self._session_id:
            self._new_session(raw_data)
            hiura_msg = self._hiura_data.get("chat_msg", "")
            if hiura_msg:
                self._push("ai1", "Hiura", hiura_msg, 1)

        sh = self._hiura_data.get("sh_since_bos", 0)
        sl = self._hiura_data.get("sl_before_bos", 0)
        bias = self._hiura_data.get("bias", "neutral")
        m5_dir = "bearish" if bias == "bullish" else "bullish"

        result = senanan_idm_hunt(
            self.clients[1], self.model_ai2,
            raw_data, sh, sl, m5_dir, bias,
            self._full_ctx(),
            prompt_ctx=self.prompts.build_context('senanan')
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
            # Cek apakah MSS wajib (dari logic_rules.json)
            import json as _j
            try:
                with open("data/logic_rules.json") as _f:
                    _lc = _j.load(_f)
                _mss_req = _lc.get("entry", {}).get("mss_required", True)
            except Exception:
                _mss_req = True

            if _mss_req:
                self._phase = "bos_guard"
                logger.info(f"[SENANAN] IDM @ {result.get('watch_level',0):.4f} | Fase → bos_guard (tunggu MSS)")
            else:
                # MSS tidak wajib — saat IDM disentuh langsung ke entry_sniper
                self._phase = "bos_guard"  # tetap bos_guard untuk tunggu IDM disentuh
                logger.info(f"[SENANAN] IDM @ {result.get('watch_level',0):.4f} | Fase → bos_guard (IDM touch = entry)")
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
            bias, sh, sl,
            prompt_ctx=self.prompts.build_context('shina')
        )
        self._shina_data = result
        self._push("ai3", "Shina", result.get("chat_msg", ""), 3)

        decision = result.get("decision", "wait")

        # Cek mss_required dari logic_rules
        import json as _jj
        try:
            with open("data/logic_rules.json") as _ff:
                _lcc = _jj.load(_ff)
            _mss_req2 = _lcc.get("entry", {}).get("mss_required", True)
        except Exception:
            _mss_req2 = True

        if decision == "entry" or (not _mss_req2 and triggered_items):
            # IDM disentuh dan MSS tidak wajib → langsung entry
            self._phase = "entry_sniper"
            logger.info(f"[SHINA] {'MSS terkonfirmasi' if decision=='entry' else 'IDM touched, MSS tidak wajib'} → entry sniper")

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
            trade_mem, self._full_ctx(),
            prompt_ctx=self.prompts.build_context('yusuf')
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

                # Guard: pastikan AI tidak naikkan min_confidence melebihi batas atas
                max_conf_allowed = self.rules.get("entry", "max_confidence_allowed", default=0.85)
                if min_conf > max_conf_allowed:
                    logger.warning(f"[YUSUF] min_confidence {min_conf:.0%} melebihi max_confidence_allowed {max_conf_allowed:.0%} — di-clamp")
                    self.rules.rules.setdefault("entry", {})["min_confidence"] = max_conf_allowed
                    self.rules._save(self.rules.rules)

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

    def _handle_user_chat(self, raw_data: dict):
        """Cek apakah ada pesan user yang perlu dijawab Katyusha."""
        if not self._katyusha_key:
            return
        try:
            import requests as _req
            port = int(os.environ.get("PORT", 8080))
            resp = _req.get(f"http://localhost:{port}/api/chat/pending", timeout=3)
            pending = resp.json()
            if not pending:
                return

            # Ada pesan user — Katyusha jawab
            user_msg = pending[-1].get("content", "")
            logger.info(f"[CHAT] User: {user_msg[:60]}")

            # Siapkan konteks bot
            price    = raw_data.get("price", 0)
            stats    = self.memory.get_stats()
            balance  = self.executor.get_account_balance()
            min_rr   = self.rules.tp_min_rr
            phase    = self._phase

            # Baca langsung dari file — pastikan data fresh
            def _rjson(path):
                try:
                    with open(path, encoding="utf-8") as _f:
                        d = json.load(_f)
                    return {k:v for k,v in d.items() if not k.startswith("_")}
                except Exception:
                    return {}
            # Full JSON untuk Katyusha — tidak dipotong
            rules_summary   = json.dumps(_rjson("data/rules.json"),       ensure_ascii=False, separators=(',',':'))
            logic_summary   = json.dumps(_rjson("data/logic_rules.json"), ensure_ascii=False, separators=(',',':'))
            prompts_summary = json.dumps(_rjson("data/prompts.json"),      ensure_ascii=False, separators=(',',':'))

            watchlist_text = self.watchlist.summary()

            # Ambil history chat untuk memori percakapan
            try:
                hist_resp = _req.get(f"http://localhost:{port}/api/chat", timeout=3)
                chat_history = hist_resp.json() if hist_resp.ok else []
            except Exception:
                chat_history = []

            # Bangun messages dengan history (max 20 pesan terakhir)
            messages = [
                {
                    "role": "system",
                    "content": f"""Kamu adalah Katyusha, supervisor ICT trading bot dengan authority penuh.
Kamu bisa mengubah rules.json, logic_rules.json, dan prompts.json secara langsung.

STATUS BOT SAAT INI:
- Symbol: {self.symbol} | Harga: {price} | Fase: {phase}
- Saldo USDT: {balance:.2f}
- Total trade: {stats.get('total_trades',0)} | Win rate: {stats.get('win_rate',0):.0%}
- Watchlist: {watchlist_text}

RULES LENGKAP (data/rules.json):
{rules_summary}

LOGIC LENGKAP (data/logic_rules.json):
{logic_summary}

PROMPTS LENGKAP (data/prompts.json):
{prompts_summary}

KEMAMPUANMU:
- Jika owner minta ubah rules/logic/prompts → langsung apply dengan format JSON di akhir response:
  <APPLY_CHANGES>
  {{"rules_changes": [{{"section": "entry", "key": "min_confidence", "new": 0.65, "reason": "..."}}],
    "logic_changes": [],
    "prompts_changes": [{{"ai": "hiura", "key": "focus", "new": "...", "reason": "..."}}]}}
  </APPLY_CHANGES>
- Jika hanya jawab pertanyaan → tidak perlu blok APPLY_CHANGES
- Kamu PUNYA MEMORI — ini adalah history chat kita
- Gaya: santai, informatif, bahasa Indonesia, max 4 kalimat"""
                }
            ]

            # Tambah history (skip pesan pertama yang mungkin kosong)
            for m in chat_history[-20:]:
                role = m.get("role", "user")
                txt  = m.get("content", "")
                if txt and role in ("user", "katyusha"):
                    messages.append({
                        "role": "user" if role == "user" else "assistant",
                        "content": txt
                    })

            # Pesan user saat ini (sudah ada di history, tapi pastikan ada)
            if not messages or messages[-1].get("content") != user_msg:
                messages.append({"role": "user", "content": user_msg})

            import requests
            k_resp = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {self._katyusha_key}",
                    "Content-Type": "application/json",
                    "HTTP-Referer": "https://bot-botan-ict.railway.app",
                },
                json={
                    "model": "anthropic/claude-sonnet-4-6",
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 800,
                },
                timeout=30,
            )
            answer = k_resp.json()["choices"][0]["message"]["content"].strip()
            logger.info(f"[CHAT] Katyusha: {answer[:100]}")

            # Parse dan apply perubahan kalau ada blok APPLY_CHANGES
            import re as _re
            apply_match = _re.search(r"<APPLY_CHANGES>(.*?)</APPLY_CHANGES>", answer, _re.DOTALL)
            clean_answer = _re.sub(r"<APPLY_CHANGES>.*?</APPLY_CHANGES>", "", answer, flags=_re.DOTALL).strip()
            if apply_match:
                raw_json = apply_match.group(1).strip()
                logger.info(f"[CHAT] APPLY_CHANGES: {raw_json[:300]}")
                try:
                    # Hapus log lines yang mungkin masuk di tengah JSON
                    clean_lines = [l for l in raw_json.splitlines()
                                   if not any(x in l for x in ["GET /","POST /","HTTP/1.1","| INFO |","| WARNING |"])]
                    changes = json.loads("\n".join(clean_lines))

                    # Apply rules + logic + prompt_updates
                    self._katyusha_apply_changes(changes)

                    # Apply prompts_changes (format dari user chat)
                    prompts = self.prompts.prompts
                    p_dirty = False
                    for ch in changes.get("prompts_changes", []):
                        ai_k = ch.get("ai","").lower()
                        fld  = ch.get("key","")
                        val  = ch.get("new")
                        if ai_k and fld and val is not None:
                            prompts.setdefault(ai_k, {})[fld] = val
                            p_dirty = True
                            logger.info(f"[CHAT] prompt {ai_k}.{fld} = {str(val)[:60]}")
                    if p_dirty:
                        prompts["_update_reason"] = "Katyusha via chat"
                        prompts["_version"] = prompts.get("_version",1) + 1
                        self.prompts.save(prompts)

                    n = (len(changes.get("rules_changes",[])) + len(changes.get("rules_adds",[])) +
                         len(changes.get("logic_changes",[])) + len(changes.get("logic_adds",[])) +
                         len(changes.get("prompts_changes",[])))
                    logger.info(f"[CHAT] Applied {n} perubahan")
                    clean_answer += f"\n✅ {n} perubahan berhasil disimpan ke file JSON!"

                except json.JSONDecodeError as je:
                    logger.error(f"[CHAT] JSON parse error: {je} | raw: {raw_json[:150]}")
                    clean_answer += "\n⚠️ Format JSON salah, perubahan gagal diparse."
                except Exception as apply_err:
                    logger.error(f"[CHAT] Gagal apply: {apply_err}")
                    clean_answer += f"\n⚠️ Gagal apply: {str(apply_err)[:80]}"
            else:
                # Deteksi kalau Katyusha klaim sudah ubah tapi tidak ada blok
                if any(x in answer.lower() for x in ["sudah diubah","sudah saya ubah","berhasil diubah","telah diubah","saya ubah"]):
                    logger.warning("[CHAT] Katyusha klaim ubah tapi tidak ada APPLY_CHANGES block!")
                    clean_answer += "\n⚠️ Katyusha tidak menyertakan blok perubahan. Coba tanya lagi: 'ubah sekarang dengan format APPLY_CHANGES'"

            # Push jawaban ke API
            _req.post(
                f"http://localhost:{port}/api/chat/answer",
                json={"answer": clean_answer},
                timeout=3,
            )

            # Juga push ke sesi chat grup kalau ada
            if self._session_id:
                self._push("katyusha", "Katyusha", clean_answer, 5)

        except Exception as e:
            logger.warning(f"[CHAT] Error handling user chat: {str(e)[:80]}")

    def _monitor_trades(self, raw_data: dict):
        """Cek trade yang sedang berjalan. Kalau ada yang close, proses debrief."""
        closed = self.executor.check_closed_trades()
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
                            self.rules.rules["_update_reason"] = f"Katyusha override: {k_post.get('summary','')}"
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
                            self.logic.rules["_update_reason"] = f"Katyusha override: {k_post.get('summary','')}"
                            self.logic.rules["_version"] = self.logic.rules.get("_version",1) + 1
                            self.logic._save(self.logic.rules)
                except Exception as e:
                    logger.error(f"[DEBRIEF ERROR] {e}")

    # ── Main Loop ────────────────────────────────────────

    async def run(self):
        logger.info("=" * 60)
        logger.info(f"BotCore starting | {self.symbol} | {'PAPER' if self.paper else 'LIVE'}")
        logger.info("=" * 60)

        # Stagger delay: hindari semua symbol hit Groq API bersamaan
        if self._stagger_delay > 0:
            logger.info(f"[{self.symbol}] Stagger delay {self._stagger_delay:.0f}s sebelum mulai...")
            await asyncio.sleep(self._stagger_delay)

        while True:
            try:
                # 1. Ambil data mentah
                raw = self.data.get_raw()
                if not raw:
                    logger.error("[BOT] Gagal ambil data market")
                    await asyncio.sleep(30)  # retry lebih cepat kalau error
                    continue

                price = raw["price"]

                # 2. Cek watchlist (Python cek harga, tidak ada analisis)
                triggered = self.watchlist.check(price, self._prev_price)
                if triggered:
                    for t in triggered:
                        logger.info(f"[TRIGGER] {t['condition'].upper()} @ {t['level']:.2f} | {t['reason'][:60]}")

                # 3. Katyusha review setiap 1 jam
                import time as _time
                now_ts = _time.time()
                if (self._katyusha_key and
                        now_ts - self._last_katyusha_ts >= self._katyusha_interval and
                        self._phase != "h1_scan"):
                    self._last_katyusha_ts = now_ts
                    logger.info("[KATYUSHA] Waktunya review 1 jam...")
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
                    phase_before = self._phase
                    self._run_h1_scan(raw)
                    # Kalau Hiura baru saja set fase ke fvg_wait, reset triggered
                    # supaya watchlist lama yang kebetulan trigger tidak diteruskan
                    if self._phase != phase_before:
                        triggered = []

                elif self._phase == "fvg_wait":
                    self._run_fvg_wait(raw, triggered)

                elif self._phase == "bos_guard":
                    if triggered:
                        self._run_bos_guard(raw, triggered)
                    else:
                        active = self.watchlist.get_active()
                        logger.info(f"[BOS GUARD] {price} | tunggu IDM touch | watchlist: {len(active)} level")

                elif self._phase == "entry_sniper":
                    self._run_entry_sniper(raw)

                # 4. Cek pesan user untuk Katyusha
                self._handle_user_chat(raw)

                # 5. Monitor trade aktif
                self._monitor_trades(raw)

                # 5. Update watchlist ke API
                api_server.update_watchlist(self.watchlist.to_api_dict(), self.symbol)
                api_server.update_bot_status(
                    self.symbol, self._phase, price,
                    len(self.watchlist.get_active())
                )

                self._prev_price = price

                # Stats
                stats = self.memory.get_stats()
                logger.info(
                    f"[BOT] {price} | Fase: {self._phase} | "
                    f"Watchlist: {len(self.watchlist.get_active())} level | "
                    f"Trades: {stats['total_trades']} | WR: {stats['win_rate']:.0%}"
                )

            except KeyboardInterrupt:
                logger.info("Bot dihentikan")
                break
            except Exception as e:
                logger.error(f"[BOT] Error: {e}", exc_info=True)

            await self._wait_next_m5_close()
