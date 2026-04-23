"""
StateManager — Persistent state antar agent.
Menyimpan hasil parse dari setiap agent ke data/state.json.
Ini adalah "memori bersama" semua agent dalam satu trading cycle.
"""
import json, os, logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
STATE_FILE = "data/state.json"

DEFAULT_STATE = {
    "current_bos": {
        "level": None, "direction": None, "candle_idx": None,
        "sh_since_bos": None, "sl_before_bos": None, "confirmed_at": None
    },
    "current_fvg": {
        "high": None, "low": None, "direction": None,
        "touched": False, "touch_price": None
    },
    "fvg_list": [],
    "idm_status": {
        "confirmed": False, "level": None, "watch_level": None,
        "candle_idx": None, "direction": None
    },
    "mss_status": {
        "confirmed": False, "candle_idx": None,
        "freeze_high": None, "freeze_low": None, "sl_level": None
    },
    "reset_count": 0,
    "last_updated": None,
    "active_phase": "h1_scan",
    "last_agent": None,
    "choch_warning": False,
}


class StateManager:
    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self._state = self._load()

    def _load(self) -> dict:
        if os.path.exists(STATE_FILE):
            try:
                with open(STATE_FILE) as f:
                    s = json.load(f)
                # Merge dengan default agar field baru selalu ada
                merged = {**DEFAULT_STATE}
                for k, v in s.items():
                    if isinstance(v, dict) and isinstance(merged.get(k), dict):
                        merged[k] = {**merged[k], **v}
                    else:
                        merged[k] = v
                return merged
            except Exception as e:
                logger.warning(f"[STATE] Load error: {e} — pakai default")
        return {**DEFAULT_STATE}

    def _save(self):
        self._state["last_updated"] = datetime.now(timezone.utc).isoformat()
        try:
            with open(STATE_FILE, "w") as f:
                json.dump(self._state, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"[STATE] Save error: {e}")

    @property
    def state(self) -> dict:
        return self._state

    def get(self, *keys, default=None):
        val = self._state
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default

    # ── Update dari hasil parse agent ──────────────────

    def update_from_hiura(self, parsed: dict):
        """Update state dari output Hiura (BOS + FVG)."""
        if parsed.get("bos_found") and parsed.get("bos_level"):
            self._state["current_bos"] = {
                "level":        parsed.get("bos_level"),
                "direction":    parsed.get("bos_type", parsed.get("bias", "")),
                "candle_idx":   parsed.get("bos_candle_idx"),
                "sh_since_bos": parsed.get("sh_since_bos"),
                "sl_before_bos":parsed.get("sl_before_bos"),
                "confirmed_at": datetime.now(timezone.utc).isoformat(),
            }
            self._state["fvg_list"] = parsed.get("fvg_list", [])
            self._state["choch_warning"] = parsed.get("choch_warning", False)
            # BOS baru → reset IDM dan MSS
            self._state["idm_status"] = {**DEFAULT_STATE["idm_status"]}
            self._state["mss_status"] = {**DEFAULT_STATE["mss_status"]}
            self._state["reset_count"] = 0
            logger.info(f"[STATE] BOS update: {self._state['current_bos']['direction']} @ {self._state['current_bos']['level']}")
        self._state["last_agent"] = "hiura"
        self._save()

    def update_from_senanan(self, parsed: dict):
        """Update state dari output Senanan (IDM)."""
        if parsed.get("idm_found"):
            self._state["idm_status"] = {
                "confirmed":   True,
                "level":       parsed.get("candle_a_level", parsed.get("idm_level")),
                "watch_level": parsed.get("watch_level"),
                "candle_idx":  parsed.get("candle_a_idx"),
                "direction":   parsed.get("idm_direction", parsed.get("direction")),
            }
            logger.info(f"[STATE] IDM confirmed: watch={self._state['idm_status']['watch_level']}")
        self._state["last_agent"] = "senanan"
        self._save()

    def update_from_shina(self, parsed: dict):
        """Update state dari output Shina (MSS/BOS M5)."""
        decision = parsed.get("decision", "wait")
        if decision == "entry":
            self._state["mss_status"] = {
                "confirmed":   True,
                "candle_idx":  parsed.get("mss_candle_idx"),
                "freeze_high": parsed.get("freeze_high", parsed.get("mss_candle_high")),
                "freeze_low":  parsed.get("freeze_low",  parsed.get("mss_candle_low")),
                "sl_level":    parsed.get("sl_level", parsed.get("mss_candle_low" if "buy" in str(parsed.get("direction","")) else "mss_candle_high")),
            }
            logger.info(f"[STATE] MSS confirmed: freeze={self._state['mss_status']['freeze_low']}–{self._state['mss_status']['freeze_high']}")
        elif decision == "reset_idm":
            self._state["idm_status"] = {**DEFAULT_STATE["idm_status"]}
            self._state["mss_status"] = {**DEFAULT_STATE["mss_status"]}
            self._state["reset_count"] += 1
            logger.info(f"[STATE] IDM reset #{self._state['reset_count']}")
        self._state["last_agent"] = "shina"
        self._save()

    def update_from_yusuf(self, parsed: dict):
        """Update state dari output Yusuf (entry decision)."""
        self._state["last_agent"] = "yusuf"
        self._save()

    def update_phase(self, phase: str):
        self._state["active_phase"] = phase
        self._save()

    def increment_reset(self):
        self._state["reset_count"] += 1
        self._save()
        return self._state["reset_count"]

    def full_reset(self, reason: str = ""):
        """Reset seluruh state trading (BOS baru atau CHOCH)."""
        logger.info(f"[STATE] Full reset: {reason}")
        self._state = {**DEFAULT_STATE}
        self._save()

    def to_context_str(self) -> str:
        """Return state sebagai string konteks untuk AI."""
        s = self._state
        bos  = s["current_bos"]
        idm  = s["idm_status"]
        mss  = s["mss_status"]
        lines = [
            f"=== TRADING STATE ===",
            f"Phase: {s['active_phase']} | Reset count: {s['reset_count']}",
            f"BOS: {bos['direction']} @ {bos['level']} | SH={bos['sh_since_bos']} SL={bos['sl_before_bos']}",
            f"IDM: {'✓ confirmed' if idm['confirmed'] else '✗ not confirmed'} | watch={idm['watch_level']}",
            f"MSS: {'✓ confirmed' if mss['confirmed'] else '✗ not confirmed'} | freeze={mss['freeze_low']}–{mss['freeze_high']}",
            f"FVG count: {len(s['fvg_list'])} | CHOCH warning: {s['choch_warning']}",
        ]
        return "\n".join(lines)
