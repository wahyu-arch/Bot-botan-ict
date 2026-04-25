"""
BotCore — Loop utama. AI-driven tapi hemat token.
Pre-filter Python → Call AI hanya saat struktur valid.
"""
import asyncio
import os
import json
import logging
from datetime import datetime, timezone
from groq import Groq
from data_provider import DataProvider
from ai_analysts import hiura_h1_analysis, senanan_idm_hunt, shina_bos_mss, yusuf_entry
from watchlist_engine import WatchlistEngine
from state_manager import StateManager
from rules_engine import RulesEngine
from logic_engine import LogicEngine
from prompt_engine import PromptEngine
from candle_replay import ReplayEngine
from memory_system import MemorySystem
from risk_manager import RiskManager
from trade_executor import TradeExecutor
from ict_analyzer import ICTAnalyzer
import api_server

logger = logging.getLogger(__name__)
os.makedirs("logs", exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                    handlers=[logging.StreamHandler(), logging.FileHandler("logs/bot.log")])

class BotCore:
    def __init__(self, symbol: str = ""):
        self.symbol = symbol or os.getenv("TRADING_SYMBOL", "BTCUSDT").strip()
        self.symbols = [self.symbol]
        self.paper = os.getenv("PAPER_TRADING", "true").lower() == "true"
        self.scan_sec = int(os.getenv("SCAN_INTERVAL_SECONDS", "0"))

        # Groq Clients & Qwen Models
        groq_key = os.environ.get("GROQ_API_KEY_AI1") or os.environ.get("GROQ_API_KEY")
        if not groq_key: raise EnvironmentError("GROQ_API_KEY tidak ditemukan!")
        self.client = Groq(api_key=groq_key)
        self.model = os.getenv("GROQ_MODEL_QWEN", "qwen-2.5-72b")  # Ganti ke qwen-2.5-32b jika rate limit

        # Components
        self.data = DataProvider(self.symbol)
        self.analyzer = ICTAnalyzer()
        self.watchlist = WatchlistEngine()
        self.state = StateManager()
        self.rules = RulesEngine()
        self.logic = LogicEngine()
        self.prompts = PromptEngine()
        self.memory = MemorySystem()
        self.risk = RiskManager()
        self.executor = TradeExecutor(paper_mode=self.paper)
        self.replay = ReplayEngine()

        self.phase = "h1_scan"
        self._hiura_ctx = {}
        self._senanan_ctx = {}
        self._shina_ctx = {}
        self._last_bos = 0.0
        self._pending_msg = ""

        # Phase Router
        self.PHASE_ROUTER = {
            "h1_scan": self._run_h1_scan,
            "fvg_wait": self._run_fvg_wait,
            "bos_guard": self._run_bos_guard,
            "entry_sniper": self._run_entry_sniper,
        }
        logger.info(f"BotCore ready | {self.symbol} | Qwen={self.model} | Paper={self.paper}")

    def _full_ctx(self) -> str:
        return json.dumps({"rules": self.rules.rules, "logic": self.logic.rules, "state": self.state.state}, ensure_ascii=False)

    def _handoff_ctx(self, from_ai: str) -> str:
        if from_ai == "hiura": return json.dumps({"bias": self._hiura_ctx.get("bias"), "bos_level": self._hiura_ctx.get("bos_level")})
        if from_ai == "senanan": return json.dumps({"idm_level": self._senanan_ctx.get("watch_level")})
        return ""

    def _reset(self, reason=""):
        self.phase = "h1_scan"
        self._hiura_ctx = self._senanan_ctx = self._shina_ctx = {}
        self._last_bos = 0.0
        self.watchlist.clear_untriggered()
        self.state.full_reset(reason)
        logger.info(f"[RESET] {reason}")

    async def _wait_m5(self):
        if self.scan_sec > 0:
            await asyncio.sleep(self.scan_sec)
            return
        now = datetime.now(timezone.utc)
        wait = (5 - (now.minute % 5)) * 60 - now.second + 2
        await asyncio.sleep(max(wait, 5))

    def _run_h1_scan(self, raw):
        # PRE-FILTER: Cek Python dulu, hemat API
        ctx = self.analyzer.quick_check(raw)
        if not ctx["h1_bos"]:
            self._pending_msg = f"Hiura: Scan H1... {raw['price']}"
            return
        # BOS ditemukan → panggil AI-1
        res = hiura_h1_analysis(self.client, self.model, raw, self._full_ctx())
        if not res or not res.get("bos_found"):
            logger.info("[HIURA] AI tidak confirm BOS → skip")
            return
        self._hiura_ctx = res
        self._pending_msg = res.get("chat_msg", "Hiura: BOS H1 aktif")
        self._last_bos = res.get("bos_level", 0)
        self.state.update_from_hiura(res)

        # Inject watchlist FVG
        for fvg in res.get("fvg_list", [])[:2]:
            if not fvg.get("filled"):
                self.watchlist.add(level=(fvg["high"]+fvg["low"])/2, condition="touch", reason="FVG touch", phase="fvg_wait")
        self.phase = "fvg_wait"

    def _run_fvg_wait(self, raw, triggers):
        if not triggers: return
        for t in triggers:
            if t.get("phase") == "fvg_wait":
                logger.info(f"[FVG] Disentuh @ {t['level']} → Call Senanan")
                bias = self._hiura_ctx.get("bias", "neutral")
                m5_dir = "bearish" if bias == "bullish" else "bullish"
                sh, sl = self._hiura_ctx.get("sh", 0), self._hiura_ctx.get("sl", 0)
                res = senanan_idm_hunt(self.client, self.model, raw, sh, sl, m5_dir, bias, self._full_ctx(), self._handoff_ctx("hiura"))
                if not res or not res.get("idm_found"):
                    logger.info("[SENANAN] IDM tidak ditemukan → stay fvg_wait")
                    return
                self._senanan_ctx = res
                for wl in res.get("watchlist", []):
                    self.watchlist.add(level=wl["level"], condition=wl.get("condition","touch"), assigned_to=wl.get("assigned_to","shina"), phase="bos_guard")
                self.phase = "bos_guard"
                return

    def _run_bos_guard(self, raw, triggers):
        if not triggers: return
        logger.info("[IDM] Disentuh → Call Shina")
        res = shina_bos_mss(self.client, self.model, raw, self._senanan_ctx, self._hiura_ctx.get("bias","neutral"),
                            self._hiura_ctx.get("sh",0), self._hiura_ctx.get("sl",0), self._full_ctx(), self._handoff_ctx("senanan"))
        if not res: return
        self._shina_ctx = res
        dec = res.get("decision", "wait")
        if dec == "entry":
            self.phase = "entry_sniper"
        elif dec == "reset_idm":
            self._senanan_ctx = self._shina_ctx = {}
            self.watchlist.clear_untriggered()
            self.phase = "fvg_wait"
        else:
            for wl in res.get("watchlist", []):
                self.watchlist.add(level=wl["level"], condition=wl.get("condition"), phase="bos_guard")

    def _run_entry_sniper(self, raw):
        mem = self.memory.get_recent_trades(3)
        res = yusuf_entry(self.client, self.model, raw, self._hiura_ctx, self._shina_ctx, mem, self._full_ctx(), self._handoff_ctx("shina"))
        if not res or res.get("decision") != "entry":
            logger.info(f"[YUSUF] Skip: {res.get('skip_reason','invalid')}")
            self._reset("cycle_skip")
            return
        # Eksekusi
        e, s, t, c = res.get("entry",0), res.get("sl",0), res.get("tp",0), res.get("confidence",0)
        if abs(e-s) > 1e-5 and abs(t-e) > 1e-5 and c >= 0.6:
            qty = self.risk.calculate_lot_size(entry=e, stop_loss=s, symbol=self.symbol)
            self.executor.execute(direction=res.get("direction","buy"), entry_price=e, stop_loss=s, take_profit=t, lot_size=qty)
            logger.info(f"[ENTRY] {res['direction']} @ {e:.4f} SL={s:.4f} TP={t:.4f}")
        self._reset("trade_done")

    async def run(self):
        logger.info("="*50)
        while True:
            try:
                raw = self.data.get_raw()
                if not raw: await self._wait_m5(); continue
                price = raw["price"]
                triggers = self.watchlist.check(price)

                # Dynamic dispatch
                handler = self.PHASE_ROUTER.get(self.phase)
                if handler:
                    handler(raw, triggers)

                # Push live msg
                if self._pending_msg:
                    api_server.push_live_msg("ai1", "Hiura", self._pending_msg, self.symbol)
                    self._pending_msg = ""

                api_server.update_bot_status(self.symbol, self.phase, price, len(self.watchlist.get_active()))
                await self._wait_m5()
            except KeyboardInterrupt: break
            except Exception as e:
                logger.error(f"[BOT] {e}", exc_info=True)
                await asyncio.sleep(10)