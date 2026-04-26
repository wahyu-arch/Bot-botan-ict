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
from state_manager import StateManager
from rules_engine import RulesEngine
from logic_engine import LogicEngine
from prompt_engine import PromptEngine
from candle_replay import ReplayEngine, format_replay_for_ai
from memory_system import MemorySystem
from risk_manager import RiskManager
from trade_executor import TradeExecutor
import api_server
from api_server import is_katyusha_enabled

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
        # Distribusi model — tiap AI pakai model berbeda agar limit Groq tidak shared
        # Qwen QwQ 32B (thinking) untuk Shina — paling butuh deep reasoning
        # Model ringan untuk Hiura/Senanan — lebih sering dipanggil
        # Default bisa di-override via env vars di Railway
        self.model_main = os.getenv("GROQ_MODEL_MAIN", "qwen-qwq-32b")           # Hiura — H1 analyst
        self.model_ai1  = os.getenv("GROQ_MODEL_AI1",  "qwen-qwq-32b")           # Senanan — IDM hunter
        self.model_ai2  = os.getenv("GROQ_MODEL_AI2",  "qwen-qwq-32b")           # Shina — BOS/MSS (paling butuh thinking)
        self.model_ai3  = os.getenv("GROQ_MODEL_AI3",  "qwen-qwq-32b")           # Yusuf — entry sniper
        self.model_json = os.getenv("GROQ_MODEL_JSON",  "qwen-qwq-32b")          # fallback JSON

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
        self.watchlist  = WatchlistEngine()
        self.state_mgr  = StateManager()
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
        self._pending_hiura_msg = ""
        self._prev_price    = 0.0
        self._last_bos_lvl  = 0.0
        self._cycles_in_phase = 0        # Python Fix 3: stuck detection
        self._last_phase_change = ""     # timestamp phase change terakhir
        self._pending_notifications: dict = {}

        # Data dari tiap AI (disimpan antar fase)
        self._hiura_data  : dict = {}
        self._senanan_data: dict = {}
        self._shina_data  : dict = {}

        # Python Fix 3: max cycles per phase sebelum auto-reset
        self._MAX_CYCLES_PER_PHASE = {
            "fvg_wait":      10,
            "idm_hunt":      15,
            "bos_guard":     10,
            "entry_sniper":  5,
        }

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
        """Kirim pesan ke live feed (selalu) dan ke sesi diskusi (kalau ada)."""
        if not msg:
            return
        # Live feed: selalu tampil di HTML
        api_server.push_live_msg(ai_key, name, msg, self.symbol)
        # Sesi diskusi: hanya saat loss debrief
        if self._session_id:
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
        self._last_bos_lvl  = 0.0
        self.watchlist.clear_untriggered()
        if self._session_id:
            self._finish_session({"consensus": reason or "reset"})
        self._session_id = ""
        self._pending_hiura_msg = ""
        # Sync state
        self.state_mgr.full_reset(reason)
        self.state_mgr.update_phase("h1_scan")

    def _parse_update_self(self, ai_name: str, result: dict):
        """
        UPDATE_SELF: setiap AI bisa update extra_instructions-nya sendiri
        berdasarkan apa yang dia pelajari dari analisis siklus ini.

        Format di response AI (di dalam _raw):
        <UPDATE_SELF>
        {
          "extra_instructions": "insight baru yang harus saya ingat...",
          "append": true
        }
        </UPDATE_SELF>

        append=true → ditambahkan ke instruksi existing (tidak replace)
        append=false → replace seluruh extra_instructions
        """
        import re as _re
        raw = result.get("_raw", "")
        if not raw or "<UPDATE_SELF>" not in raw:
            return

        has_close = "</UPDATE_SELF>" in raw
        if not has_close:
            logger.warning(f"[{ai_name.upper()}] UPDATE_SELF tag tidak closed — skip")
            return

        m = _re.search(r"<UPDATE_SELF>(.*?)</UPDATE_SELF>", raw, _re.DOTALL)
        if not m:
            return

        try:
            data = json.loads(m.group(1).strip())
        except Exception as e:
            logger.warning(f"[{ai_name.upper()}] UPDATE_SELF JSON invalid: {e}")
            return

        new_instr = data.get("extra_instructions", "").strip()
        if not new_instr:
            return

        append = data.get("append", True)
        ai_key = ai_name.lower()

        current = self.prompts.prompts.get(ai_key, {}).get("extra_instructions", "")
        if append and current:
            # Tambahkan sebagai paragraph baru, bukan replace
            updated = current.rstrip() + "\n\n[Update " + __import__("datetime").datetime.now().strftime("%m/%d %H:%M") + "]:\n" + new_instr
        else:
            updated = new_instr

        self.prompts.prompts.setdefault(ai_key, {})["extra_instructions"] = updated
        self.prompts.save(self.prompts.prompts)
        logger.info(f"[{ai_name.upper()}] UPDATE_SELF: extra_instructions diupdate ({len(new_instr)} chars, append={append})")
        self._push(f"ai_{ai_key}", ai_name.capitalize(),
                   f"💭 [{ai_name.capitalize()}] updated own memory: {new_instr[:80]}...", 0)

    def _parse_update_hiura(self, result: dict):
        """FIX 2+3: Parse UPDATE_HIURA dari output Hiura, auto-write prompts + inject watchlist."""
        import re as _re
        raw = result.get("_raw", "")
        if not raw:
            return
        has_open  = "<UPDATE_HIURA>" in raw
        has_close = "</UPDATE_HIURA>" in raw
        if not (has_open and has_close):
            if has_open:
                logger.warning("[HIURA] UPDATE_HIURA tag tidak closed — skip")
            return
        m = _re.search(r"<UPDATE_HIURA>(.*?)</UPDATE_HIURA>", raw, _re.DOTALL)
        if not m:
            return
        try:
            data = json.loads(m.group(1).strip())
        except Exception as e:
            logger.warning(f"[HIURA] UPDATE_HIURA JSON invalid: {e}")
            return

        # Auto-write extra_instructions ke prompts.json
        try:
            fvg_zone = data.get("fvg_zone", {})
            new_instr = (
                f"Harga sekarang {data.get('current_price','')}. "
                f"BOS {data.get('bos_direction','')} H1 aktif — level {data.get('bos_level','')}. "
                f"SH aktif = {data.get('sh','')}. SL target = {data.get('sl','')}. "
                f"FVG aktif: gap {fvg_zone.get('low','')}–{fvg_zone.get('high','')}, "
                f"mid {fvg_zone.get('mid','')}. "
                f"FVG in range: {data.get('fvg_in_range','')}."
            )
            self.prompts.prompts.setdefault("hiura", {})["extra_instructions"] = new_instr
            self.prompts.save(self.prompts.prompts)
            logger.info(f"[HIURA] UPDATE_HIURA: prompts.json updated")
        except Exception as e:
            logger.warning(f"[HIURA] Gagal update prompts: {e}")

        # Auto-inject watchlist dari UPDATE_HIURA
        wl_items = data.get("watchlist", [])
        injected = 0
        for item in wl_items:
            lvl = float(item.get("level", 0))
            if lvl <= 0:
                continue
            cond = item.get("condition", "touch")
            # Cek duplikat
            existing = [w for w in self.watchlist.get_active()
                        if abs(w["level"] - lvl) < 1e-9 and w["condition"] == cond]
            if existing:
                continue
            self.watchlist.add(
                level=lvl,
                condition=cond,
                reason=item.get("reason", "UPDATE_HIURA inject"),
                phase=item.get("phase", self._phase),
                session_ref=self._session_id or "hiura",
                symbol=self.symbol,
                assigned_to=item.get("assigned_to", ""),
                action=item.get("action", ""),
                ttl_hours=item.get("ttl_hours", 24.0),
            )
            injected += 1
        if injected:
            logger.info(f"[HIURA] UPDATE_HIURA: {injected} watchlist item injected")

        # Update state dari UPDATE_HIURA
        if data.get("bos_level"):
            self.state_mgr.update_from_hiura({
                "bos_found": True,
                "bos_level": data.get("bos_level"),
                "bos_type": data.get("bos_direction",""),
                "sh_since_bos": data.get("sh"),
                "sl_before_bos": data.get("sl"),
                "fvg_list": data.get("watchlist", []),
            })

        # FIX 4: Handoff — baca next_phase dari UPDATE_HIURA
        next_phase = data.get("next_phase", "")
        if next_phase and next_phase in ("fvg_wait","idm_hunt","bos_guard","entry_sniper"):
            logger.info(f"[HIURA] next_phase dari UPDATE_HIURA: {next_phase}")
            self._phase = next_phase
            self.state_mgr.update_phase(next_phase)

    # ── Allowed actions per AI (Katyusha spec) ──────────────
    _AI_AUTHORITY = {
        "hiura":    {"force_phase": {"fvg_wait","idm_hunt","h1_scan"},
                     "add_watchlist": True, "notify": True},
        "senanan":  {"force_phase": {"bos_guard","fvg_wait","h1_scan","idm_hunt"},
                     "add_watchlist": True, "notify": True},
        "shina":    {"force_phase": {"entry_sniper","bos_guard","idm_hunt","fvg_wait","h1_scan"},
                     "add_watchlist": True, "notify": True},
        "yusuf":    {"force_phase": {"h1_scan"},          # Yusuf bisa reset ke h1_scan setelah entry
                     "add_watchlist": True, "notify": True},
        "katyusha": {"force_phase": {"h1_scan","fvg_wait","idm_hunt","bos_guard","entry_sniper"},
                     "add_watchlist": True, "notify": True},
    }

    def _execute_actions(self, ai_name: str, actions: list, raw_data: dict):
        """
        ActionExecutor — baca field `actions` dari response AI dan eksekusi.
        Setiap AI punya authority sesuai _AI_AUTHORITY.
        """
        if not actions or not isinstance(actions, list):
            return

        authority = self._AI_AUTHORITY.get(ai_name.lower(), {})

        for act in actions:
            atype = act.get("type", "")

            # ── force_phase ────────────────────────────────────────
            if atype == "force_phase":
                phase = act.get("phase", "")
                allowed = authority.get("force_phase", set())
                if phase in allowed:
                    old = self._phase
                    self._phase = phase
                    self.state_mgr.update_phase(phase)
                    logger.info(f"[ACTION] {ai_name} force_phase: {old} → {phase}")
                    # Reset state yang relevan saat phase berubah
                    if phase in ("h1_scan",):
                        self._reset(f"{ai_name}_forced_h1scan")
                        return  # _reset sudah handle sisanya
                    elif phase in ("idm_hunt", "fvg_wait"):
                        self._senanan_data = {}
                        self._shina_data   = {}
                        self.watchlist.clear_untriggered()
                    api_server.push_live_msg(ai_name, ai_name.capitalize(),
                        f"⚡ Phase → {phase} (by {ai_name})", self.symbol)
                else:
                    logger.warning(f"[ACTION] {ai_name} tidak punya authority force_phase={phase}")

            # ── add_watchlist ───────────────────────────────────────
            elif atype == "add_watchlist":
                if not authority.get("add_watchlist"):
                    continue
                lvl = float(act.get("level", 0))
                if lvl <= 0:
                    logger.warning(f"[ACTION] {ai_name} add_watchlist level=0 — skip")
                    continue
                cond        = act.get("condition", "touch")
                assigned_to = act.get("assigned_to", "")
                action_tag  = act.get("action", "")
                reason      = act.get("reason", f"{ai_name} watchlist")
                phase_tag   = act.get("phase", self._phase)
                ttl         = float(act.get("ttl_hours", 24.0))

                # Cek duplikat
                existing = [w for w in self.watchlist.get_active()
                            if abs(w["level"] - lvl) < 1e-9 and w["condition"] == cond]
                if existing:
                    logger.debug(f"[ACTION] Watchlist duplikat @ {lvl} — skip")
                    continue

                self.watchlist.add(
                    level=lvl, condition=cond, reason=reason,
                    phase=phase_tag, session_ref=self._session_id or ai_name,
                    symbol=self.symbol, assigned_to=assigned_to,
                    action=action_tag, ttl_hours=ttl,
                )
                logger.info(f"[ACTION] {ai_name} add_watchlist: {cond} @ {lvl} → {assigned_to or 'unassigned'}")

            # ── notify ──────────────────────────────────────────────
            elif atype == "notify":
                if not authority.get("notify"):
                    continue
                to_ai   = act.get("to", "")
                message = act.get("message", "")
                if not message:
                    continue
                # Inject notifikasi ke context AI yang dituju via pending_messages
                if not hasattr(self, "_pending_notifications"):
                    self._pending_notifications = {}
                self._pending_notifications.setdefault(to_ai.lower(), []).append(
                    f"[Dari {ai_name}]: {message}"
                )
                logger.info(f"[ACTION] {ai_name} → notify {to_ai}: {message[:60]}")
                api_server.push_live_msg(ai_name, ai_name.capitalize(),
                    f"📨 → {to_ai.capitalize()}: {message[:80]}", self.symbol)

            else:
                logger.debug(f"[ACTION] Unknown action type: {atype}")

    def _get_notifications(self, ai_name: str) -> str:
        """Ambil notifikasi pending untuk AI ini, lalu hapus."""
        if not hasattr(self, "_pending_notifications"):
            return ""
        msgs = self._pending_notifications.pop(ai_name.lower(), [])
        return "\n".join(msgs) if msgs else ""

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
    def _logic_ctx(self) -> str:
        return self.logic.get_context_for_ai()

    # ── Python Fixes dari Katyusha ────────────────────────

    @staticmethod
    def validate_rr(entry: float, sl: float, tp: float,
                    min_rr: float = 2.0) -> tuple[bool, float]:
        """Python Fix 6: Validasi RR di Python, bukan hanya di AI."""
        risk   = abs(entry - sl)
        reward = abs(tp - entry)
        if risk == 0:
            return False, 0.0
        rr = reward / risk
        return rr >= min_rr, round(rr, 2)

    @staticmethod
    def is_ranging(h1_candles: list, lookback: int = 20,
                   threshold_pct: float = 1.5) -> bool:
        """Python Fix 9: Deteksi ranging market dari candle H1."""
        if len(h1_candles) < lookback:
            return False
        recent = h1_candles[-lookback:]
        highs  = [c["h"] for c in recent]
        lows   = [c["l"] for c in recent]
        if not lows or min(lows) == 0:
            return False
        range_pct = (max(highs) - min(lows)) / min(lows) * 100
        return range_pct < threshold_pct

    def _set_phase(self, new_phase: str, reason: str = "", by: str = "python"):
        """Python Fix 10: Phase transition dengan log lengkap."""
        if new_phase == self._phase:
            return
        old_phase = self._phase
        self._phase = new_phase
        self._cycles_in_phase = 0
        self._last_phase_change = datetime.now(timezone.utc).isoformat()
        self.state_mgr.update_phase(new_phase)

        msg = f"Phase: {old_phase} → {new_phase} | by={by}" + (f" | {reason}" if reason else "")
        logger.info(f"[PHASE] {msg}")
        api_server.push_live_msg("system", "Bot", f"⚡ {msg}", self.symbol)

        # Update state cycles counter
        self.state_mgr.state["cycles_in_phase"] = 0
        self.state_mgr.state["last_phase_change"] = self._last_phase_change
        self.state_mgr._save()

    def _check_stuck(self) -> bool:
        """Python Fix 3: Cek apakah bot stuck, auto-reset kalau perlu."""
        max_c = self._MAX_CYCLES_PER_PHASE.get(self._phase, 999)
        self._cycles_in_phase += 1
        self.state_mgr.state["cycles_in_phase"] = self._cycles_in_phase
        self.state_mgr._save()

        if self._cycles_in_phase > max_c:
            logger.warning(
                f"[STUCK] Phase {self._phase} sudah {self._cycles_in_phase} siklus "
                f"(max={max_c}) — auto-reset"
            )
            api_server.push_live_msg("katyusha", "Katyusha",
                f"⚠️ Stuck detection: {self._phase} {self._cycles_in_phase} siklus > {max_c} — force reset",
                self.symbol)

            # Stuck rules sesuai Katyusha spec
            if self._phase == "fvg_wait":
                self._set_phase("idm_hunt", "stuck fvg_wait", "stuck_detector")
            elif self._phase in ("bos_guard", "idm_hunt"):
                self._set_phase("idm_hunt", f"stuck {self._phase}, reset IDM", "stuck_detector")
                self._senanan_data = {}
                self._shina_data   = {}
                self.watchlist.clear_untriggered()
            elif self._phase == "entry_sniper":
                self._set_phase("h1_scan", "stuck entry_sniper, MSS expired", "stuck_detector")
                self._reset("stuck_entry_sniper")
            return True
        return False

    def _validate_entry(self, entry: float, sl: float, tp: float,
                        current_price: float) -> tuple[bool, str]:
        """Python Fix 5+6: Validasi entry sebelum order."""
        if entry <= 0 or sl <= 0 or tp <= 0:
            return False, "level 0"
        min_sl_dist = entry * 0.001
        if abs(entry - sl) < min_sl_dist:
            return False, f"SL distance terlalu kecil ({abs(entry-sl):.6f} < {min_sl_dist:.6f})"
        ok, rr = self.validate_rr(entry, sl, tp)
        if not ok:
            return False, f"RR {rr:.1f} < 2.0"
        # Cek apakah level sudah expired (harga sudah > 0.5% dari entry)
        if current_price > 0 and abs(current_price - entry) / entry > 0.005:
            return False, f"Entry expired (harga {current_price} vs entry {entry}, selisih {abs(current_price-entry)/entry*100:.1f}%)"
        return True, f"RR {rr:.1f}"

    def _full_ctx(self, replay_text: str = "", ai_name: str = "") -> dict:
        """Return semua JSON config + state + candle yang tepat per AI."""
        import json
        ctx = {
            "rules":      json.dumps({k:v for k,v in self.rules.rules.items()   if not k.startswith("_")}, ensure_ascii=False, separators=(',',':')),
            "logic":      json.dumps({k:v for k,v in self.logic.rules.items()   if not k.startswith("_")}, ensure_ascii=False, separators=(',',':')),
            "logic_raw":  {k:v for k,v in self.logic.rules.items() if not k.startswith("_")},
            "prompts":    json.dumps({k:v for k,v in self.prompts.prompts.items() if not k.startswith("_")}, ensure_ascii=False, separators=(',',':')),
            "replay_text": replay_text,
            "state":      self.state_mgr.to_context_str(),
            "state_json": json.dumps(self.state_mgr.state, ensure_ascii=False, separators=(',',':')),
            # Python Fix 2: inject hiura/senanan/shina data sebagai state antar AI
            "hiura_data":   json.dumps({k:v for k,v in self._hiura_data.items() if k != "_raw"}, ensure_ascii=False, separators=(',',':')),
            "senanan_data": json.dumps({k:v for k,v in self._senanan_data.items() if k != "_raw"}, ensure_ascii=False, separators=(',',':')),
            "shina_data":   json.dumps({k:v for k,v in self._shina_data.items() if k != "_raw"}, ensure_ascii=False, separators=(',',':')),
            # Python Fix 8: reset_count visible
            "reset_count":  self.state_mgr.get("reset_count", default=0),
            "cycles_in_phase": self._cycles_in_phase,
        }
        return ctx

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
            result["_raw"] = result.get("_raw", "")  # raw sudah di-set oleh ai_analysts
            self._parse_update_hiura(result)
            self._parse_update_self("hiura", result)  # FIX 2+3
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

        # Hanya proses BOS baru kalau fase memang h1_scan
        # Jangan interupsi fvg_wait/idm_hunt karena replay selalu menemukan BOS yang sama
        if bos_found and abs(bos_level - self._last_bos_lvl) > 1e-9 and self._phase == "h1_scan":
            self._last_bos_lvl = bos_level
            bos_type = result.get("bos_type", "")
            sh       = result.get("sh_since_bos", 0) or result.get("bos_level", 0)
            sl       = result.get("sl_before_bos", 0)
            fvgs     = result.get("fvg_list", [])

            # Problem 1: Update persistent state dari output Hiura
            self.state_mgr.update_from_hiura({
                "bos_found": True, "bos_level": bos_level, "bos_type": bos_type,
                "sh_since_bos": sh, "sl_before_bos": sl, "fvg_list": fvgs,
                "choch_warning": result.get("choch_warning", False),
            })
            # Execute actions dari Hiura
            self._execute_actions("hiura", result.get("actions", []), raw_data)
            # Auto-inject watchlist dari field watchlist (kompatibilitas)
            for wl in result.get("watchlist", []):
                if float(wl.get("level", 0)) > 0:
                    self._execute_actions("hiura", [{
                        "type": "add_watchlist", **wl
                    }], raw_data)

            # Simpan msg Hiura — akan di-push ke sesi saat FVG disentuh
            self._pending_hiura_msg = msg

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

        elif bos_found and self._phase == "h1_scan":
            # BOS sama tapi belum di-set — edge case, skip
            logger.info(f"[HIURA] BOS {bos_level} sudah ada, fase sudah {self._phase}")
        elif not bos_found:
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

        # Jalankan replay setiap siklus untuk immediate check (0 AI token)
        h1_candles = raw_data.get("h1", [])
        replay_event = self._replay.replay_h1(h1_candles, current_price=price)
        r_event = replay_event.get("event", "none")

        if r_event == "fvg_touched":
            logger.info(f"[FVG WAIT] Replay: FVG touched @ {price}")
            triggered_items = [{"phase": "fvg_wait", "level": price,
                                 "reason": "FVG touched (replay check)"}]
        elif r_event == "choch":
            msg = f"Hiura: CHOCH — {replay_event['data'].get('reason','')}. Reset."
            api_server.push_live_msg("ai1", "Hiura", msg, self.symbol)
            self._reset("choch"); self._last_bos_lvl = 0.0; return
        elif r_event == "all_fvg_filled":
            api_server.push_live_msg("ai1", "Hiura", "semua FVG filled — reset.", self.symbol)
            self._reset("all_fvg_filled"); self._last_bos_lvl = 0.0; return

        if not triggered_items:
            logger.info(f"[FVG WAIT] {price} | FVG={len(self._replay.fvgs)} | tunggu sentuh")
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

        # Pastikan pesan Hiura tertunda sudah di-push ke live feed
        pending = getattr(self, "_pending_hiura_msg", "")
        if pending:
            api_server.push_live_msg("ai1", "Hiura", pending, self.symbol)
            self._pending_hiura_msg = ""

        sh = self._hiura_data.get("sh_since_bos", 0)
        sl = self._hiura_data.get("sl_before_bos", 0)
        bias = self._hiura_data.get("bias", "neutral")
        m5_dir = "bearish" if bias == "bullish" else "bullish"

        # Inject notifikasi dari Hiura atau AI lain (notify system)
        sen_notif = self._get_notifications("senanan")
        if sen_notif:
            logger.info(f"[SENANAN] Notifikasi masuk: {sen_notif[:100]}")
            api_server.push_live_msg("ai2", "Senanan", f"📨 {sen_notif[:100]}", self.symbol)

        result = senanan_idm_hunt(
            self.clients[1], self.model_ai2,
            raw_data, sh, sl, m5_dir, bias,
            self._full_ctx(),
            prompt_ctx=self.prompts.build_context('senanan')
        )
        self._senanan_data = result
        self._push("ai2", "Senanan", result.get("chat_msg", ""), 2)
        self.state_mgr.update_from_senanan(result)
        # Execute actions dari Senanan
        self._execute_actions("senanan", result.get("actions", []), raw_data)
        self._parse_update_self("senanan", result)

        # FIX 4: baca next_phase dari output Senanan
        next_p = result.get("next_phase", "")
        if next_p and next_p in ("fvg_wait","idm_hunt","bos_guard","entry_sniper","h1_scan"):
            logger.info(f"[SENANAN] next_phase: {next_p}")
            self._phase = next_p
        self.state_mgr.update_phase(self._phase)

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
        price = raw_data.get("price", 0)

        # Cek watchlist katyusha (assigned_to="katyusha" atau "alert") — trigger alert ke chat
        katyusha_items = [t for t in triggered_items if t.get("assigned_to") in ("katyusha", "alert")]
        for kt in katyusha_items:
            alert_msg = f"⚡ Alert: harga {price} menyentuh level {kt['level']} — {kt['reason']}"
            self._push("katyusha", "Katyusha", alert_msg, 5)
            logger.info(f"[KATYUSHA ALERT] {alert_msg}")

        # Cek watchlist yang di-assign ke AI spesifik
        ai_items = [t for t in triggered_items if t.get("assigned_to") in ("hiura","senanan","shina","yusuf") and t not in katyusha_items]

        if not triggered_items or (triggered_items and not [t for t in triggered_items if t.get("phase") == "bos_guard" and not t.get("assigned_to")]):
            # Update watchlist dinamis tiap siklus — tambah level baru di sekitar harga kalau watchlist aktif < 2
            active = self.watchlist.get_active()
            bos_guard_wl = [w for w in active if w.get("phase") == "bos_guard"]
            if len(bos_guard_wl) < 2 and self._senanan_data:
                idm_watch = self._senanan_data.get("watch_level", 0)
                freeze_h  = self._shina_data.get("freeze_high", 0)
                freeze_l  = self._shina_data.get("freeze_low",  0)
                if freeze_h and freeze_l and freeze_h > freeze_l:
                    self.watchlist.add(level=freeze_h, condition="break_above", reason="freeze high — MSS bullish jika tembus",
                                       phase="bos_guard", session_ref=self._session_id, assigned_to="shina", action="check_mss")
                    self.watchlist.add(level=freeze_l, condition="break_below", reason="freeze low — BOS M5 bearish jika tembus",
                                       phase="bos_guard", session_ref=self._session_id, assigned_to="shina", action="check_mss")
            if not triggered_items or not [t for t in triggered_items if t.get("phase") == "bos_guard" and not t.get("assigned_to")]:
                logger.info(f"[BOS GUARD] {price} | tunggu IDM touch | watchlist: {len(bos_guard_wl)} level")
                return

        item = triggered_items[-1]
        logger.info(f"[BOS GUARD] IDM M5 disentuh @ {item['level']:.2f} — panggil Shina")
        shina_notif = self._get_notifications("shina")
        if shina_notif:
            logger.info(f"[SHINA] Notifikasi masuk: {shina_notif[:100]}")

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
        self.state_mgr.update_from_shina(result)
        # Execute actions dari Shina
        self._execute_actions("shina", result.get("actions", []), raw_data)
        self._parse_update_self("shina", result)

        # FIX 4+5: baca next_phase atau keyword RESET/ENTRY/WAIT dari Shina
        shina_decision = result.get("decision", result.get("next_phase", ""))
        raw_shina = result.get("_raw", "").upper()

        if shina_decision in ("reset", "reset_idm") or "RESET" in raw_shina:
            # FIX 5: auto-reset ke idm_hunt, panggil Senanan ulang
            logger.info("[SHINA] RESET — balik ke idm_hunt, Senanan dipanggil ulang")
            self.state_mgr.update_from_shina({"decision": "reset_idm"})
            self._phase = "bos_guard"  # tetap bos_guard tapi clear IDM state
            self._senanan_data = {}
            self._shina_data = {}
            self.watchlist.clear_untriggered()
            # Set watchlist baru untuk trigger Senanan lagi
            fvg_list = self._hiura_data.get("fvg_list", [])
            for fvg in [f for f in fvg_list if not f.get("filled")][:2]:
                lvl = fvg.get("mid") or (fvg["high"] + fvg["low"]) / 2
                self.watchlist.add(level=lvl, condition="touch",
                    reason="Re-hunt IDM setelah Shina reset",
                    phase="fvg_wait", session_ref=self._session_id,
                    symbol=self.symbol, assigned_to="senanan", action="re_analyze")
        elif shina_decision in ("entry",) or "ENTRY" in raw_shina:
            logger.info("[SHINA] ENTRY signal → entry_sniper")
            next_p = "entry_sniper"
            self._phase = next_p
        elif result.get("next_phase","") in ("fvg_wait","idm_hunt","bos_guard","entry_sniper"):
            self._phase = result["next_phase"]

        self.state_mgr.update_phase(self._phase)

        # Problem 4: enforce reset_count dari state
        max_resets = 3
        try:
            import json as _j
            logic_file = "data/logic_rules.json"
            if __import__("os").path.exists(logic_file):
                with open(logic_file) as _f:
                    _lr = _j.load(_f)
                max_resets = _lr.get("idm_reset", {}).get("max_reset", 3)
        except Exception:
            pass
        if self.state_mgr.get("reset_count", default=0) >= max_resets:
            logger.warning(f"[BOT] reset_count >= {max_resets} — force Hiura re-scan")
            self.state_mgr.full_reset("max_reset_exceeded")
            self._reset("max_reset_exceeded")

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
        self._parse_update_self("yusuf", result)

        decision = result.get("decision", "skip")

        if decision == "entry":
            entry = result.get("entry", 0)
            sl    = result.get("sl", 0)
            tp    = result.get("tp", 0)
            conf  = result.get("confidence", 0)
            min_conf = self.rules.entry_min_confidence

            direction = result.get("direction", "buy")

            # Python Fix 6: Validasi entry di Python (tidak hanya percaya AI)
            ok, reason = self._validate_entry(entry, sl, tp, price)
            entry_valid = ok and conf >= min_conf

            # Python Fix 9: Ranging check sebelum entry
            if entry_valid and self.is_ranging(raw.get("h1", [])):
                logger.info(f"[YUSUF] Skip — market ranging")
                entry_valid = False
                reason = "market ranging"

            if entry_valid:
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
                if not entry_valid:
                    logger.warning(
                        f"[YUSUF] Skip — entry invalid: entry={entry:.6f} sl={sl:.6f} tp={tp:.6f} "
                        f"sl_dist={sl_dist:.6f} min_sl_dist={min_sl_dist:.6f} rr={rr_check:.1f}"
                    )
                else:
                    logger.info(f"[YUSUF] Skip — conf={conf:.0%} < min={min_conf:.0%}")
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
- Jika owner minta ubah rules/logic/prompts, force phase, ATAU pasang watchlist → langsung apply:
  <APPLY_CHANGES>
  {{"rules_changes": [],
    "logic_changes": [],
    "prompts_changes": [],
    "watchlist_adds": [
      {{"level": 0.205, "condition": "break_above", "reason": "resistance level",
        "assigned_to": "shina", "action": "check_mss", "phase": "bos_guard"}},
      {{"level": 0.195, "condition": "touch", "reason": "alert harga kunci",
        "assigned_to": "katyusha", "action": "alert", "phase": "bos_guard"}}
    ],
    "watchlist_clear": false,
    "override_action": "force_phase",
    "override_phase": ""}}
  </APPLY_CHANGES>
- assigned_to pilihan: "hiura" (re-scan H1), "senanan" (cari IDM baru), "shina" (cek MSS/BOS), "yusuf" (entry), "katyusha" (alert saja)
- action pilihan: "re_analyze", "check_mss", "check_bos", "entry", "alert"
- Phase yang bisa di-force: h1_scan, fvg_wait, idm_hunt, bos_guard, entry_sniper
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

            # FIX 1: Robust parser — harus ada KEDUA tag opening dan closing
            import re as _re
            has_open  = "<APPLY_CHANGES>" in answer
            has_close = "</APPLY_CHANGES>" in answer
            if has_open and not has_close:
                logger.warning("[CHAT] APPLY_CHANGES tidak closed — skip blok, error tidak diinject ke chat")
                answer += "\n[sistem: tag APPLY_CHANGES tidak lengkap, perubahan dibatalkan]"
            apply_match = _re.search(r"<APPLY_CHANGES>(.*?)</APPLY_CHANGES>", answer, _re.DOTALL) if (has_open and has_close) else None
            clean_answer = _re.sub(r"<APPLY_CHANGES>.*?</APPLY_CHANGES>", "", answer, flags=_re.DOTALL).strip()
            # Bersihkan error message internal agar tidak bocor ke chat user
            clean_answer = _re.sub(r"\[sistem:.*?\]", "", clean_answer).strip()
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

                    # Handle force phase dari chat
                    oa = changes.get("override_action","")
                    op = changes.get("override_phase","")
                    if oa == "force_phase" and op in ("h1_scan","fvg_wait","idm_hunt","bos_guard","entry_sniper"):
                        old_p = self._phase
                        self._phase = op
                        if op == "idm_hunt":   self._replay.state = "IDM_HUNT"
                        elif op == "bos_guard": self._replay.state = "BOS_GUARD"
                        elif op == "h1_scan":
                            self._replay.reset(); self._last_bos_lvl = 0.0
                        logger.info(f"[CHAT] Katyusha force phase: {old_p} → {op}")
                        clean_answer += f"\n🎯 Phase dipindah: {old_p} → {op}"

                    # Apply watchlist additions dari Katyusha
                    wl_adds = changes.get("watchlist_adds", [])
                    for wl in wl_adds:
                        lvl = float(wl.get("level", 0))
                        if lvl > 0:
                            self.watchlist.add(
                                level=lvl,
                                condition=wl.get("condition", "touch"),
                                reason=wl.get("reason", "Katyusha watchlist"),
                                phase=wl.get("phase", self._phase),
                                session_ref=self._session_id or "katyusha",
                                assigned_to=wl.get("assigned_to", "katyusha"),
                                action=wl.get("action", "alert"),
                            )
                            logger.info(f"[KATYUSHA] +Watchlist: {wl.get('condition','touch')} @ {lvl} → assigned={wl.get('assigned_to','katyusha')}")
                    if changes.get("watchlist_clear"):
                        self.watchlist.clear_untriggered()
                        logger.info("[KATYUSHA] Watchlist dibersihkan via chat")

                    n = (len(changes.get("rules_changes",[])) + len(changes.get("rules_adds",[])) +
                         len(changes.get("logic_changes",[])) + len(changes.get("logic_adds",[])) +
                         len(changes.get("prompts_changes",[])) + len(wl_adds))
                    logger.info(f"[CHAT] Applied {n} perubahan")
                    if n > 0:
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

                # Python Fix 7: Candle freshness check
                if not self.data.is_candle_fresh():
                    logger.warning("[BOT] Candle stale — skip siklus ini")
                    api_server.push_live_msg("system","Bot","⚠️ Data candle stale, skip siklus",self.symbol)
                    await self._wait_next_m5_close()
                    continue

                # Expire watchlist kadaluarsa
                self.watchlist.expire_stale(self._phase)

                # Python Fix 3: Stuck detection
                self._check_stuck()

                # Cek watchlist trigger
                triggered = self.watchlist.check(price, self._prev_price)
                if triggered:
                    for t in triggered:
                        logger.info(f"[TRIGGER] {t['condition'].upper()} @ {t['level']} | {t['reason'][:60]}")

                # Sync phase ke state
                self.state_mgr.update_phase(self._phase)

                # 3. Katyusha review setiap 1 jam
                import time as _time
                now_ts = _time.time()
                if (self._katyusha_key and
                        is_katyusha_enabled() and
                        now_ts - self._last_katyusha_ts >= self._katyusha_interval and
                        self._phase != "h1_scan"):
                    self._last_katyusha_ts = now_ts
                    logger.info("[KATYUSHA] Waktunya review 1 jam...")
                    bot_state = {
                        "phase":     self._phase,
                        "watchlist": self.watchlist.to_api_dict(),
                        "state":     self.state_mgr.state,
                        "state_ctx": self.state_mgr.to_context_str(),
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
                    if action in ("reset",):
                        logger.info(f"[KATYUSHA] OVERRIDE: reset | {k_result.get('reasoning','')}")
                        self._last_bos_lvl = 0.0
                        self._replay.reset()
                        self._reset("katyusha_reset")

                    elif action in ("force_phase", "force_idm_hunt", "force_entry"):
                        new_phase = k_result.get("override_phase", "")
                        # Map action ke phase
                        if action == "force_idm_hunt": new_phase = "idm_hunt"
                        if action == "force_entry":     new_phase = "entry_sniper"
                        if new_phase in ("h1_scan","fvg_wait","idm_hunt","bos_guard","entry_sniper"):
                            old_phase = self._phase
                            self._phase = new_phase
                            # Kalau force ke idm_hunt, pastikan replay juga di state IDM_HUNT
                            if new_phase == "idm_hunt":
                                self._replay.state = "IDM_HUNT"
                            elif new_phase == "bos_guard":
                                self._replay.state = "BOS_GUARD"
                            elif new_phase == "h1_scan":
                                self._replay.reset()
                                self._last_bos_lvl = 0.0
                            logger.info(f"[KATYUSHA] OVERRIDE: {old_phase} → {new_phase}")
                            msg_k = f"Katyusha: force phase {old_phase} → {new_phase}. {k_result.get('reasoning','')[:80]}"
                            api_server.push_live_msg("katyusha", "Katyusha", msg_k, self.symbol)
                        else:
                            logger.warning(f"[KATYUSHA] Invalid override_phase: {new_phase}")

                    elif action == "skip_entry":
                        logger.info("[KATYUSHA] OVERRIDE: skip_entry")
                        self._reset("katyusha_skip_entry")

                    elif k_result.get("verdict") == "warning":
                        logger.warning(f"[KATYUSHA] Warning: {k_result.get('reasoning','')}")  

                # Cek watchlist assigned_to khusus — dispatch ke AI terlepas dari fase
                assigned_triggered = [t for t in triggered if t.get("assigned_to") and t.get("assigned_to") not in ("katyusha","alert","auto","")]
                for at in assigned_triggered:
                    ai_target = at.get("assigned_to", "")
                    action    = at.get("action", "alert")
                    logger.info(f"[ASSIGNED] {at['condition']} @ {at['level']} → {ai_target} ({action})")
                    if ai_target == "hiura":
                        self._run_h1_scan(raw)
                    elif ai_target == "senanan":
                        sh = self._hiura_data.get("sh_since_bos", 0)
                        sl = self._hiura_data.get("sl_before_bos", 0)
                        bias = self._hiura_data.get("bias","neutral")
                        m5_dir = "bearish" if bias == "bullish" else "bullish"
                        self._run_fvg_wait(raw, [at])  # re-trigger senanan
                    elif ai_target == "shina":
                        self._run_bos_guard(raw, [at])
                    elif ai_target == "yusuf" and action == "entry":
                        self._run_entry_sniper(raw)

                # FIX 4: Event-driven dispatch — setiap AI bisa set next_phase
                # Map phase → handler (Katyusha spec: phase_to_agent pattern)
                _phase_before = self._phase
                if self._phase == "h1_scan":
                    self._run_h1_scan(raw)
                    if self._phase != _phase_before:
                        triggered = []  # reset triggered saat fase baru

                elif self._phase == "fvg_wait":
                    self._run_fvg_wait(raw, triggered)

                elif self._phase in ("bos_guard", "idm_hunt"):
                    self._run_bos_guard(raw, triggered)

                elif self._phase == "entry_sniper":
                    self._run_entry_sniper(raw)

                # Auto-handoff: kalau phase berubah setelah dispatch, log transisi
                if self._phase != _phase_before:
                    logger.info(f"[HANDOFF] {_phase_before} → {self._phase} (otomatis)")
                    self.state_mgr.update_phase(self._phase)

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
