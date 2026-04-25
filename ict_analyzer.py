"""
ICT Analyzer - Analisis struktur market ICT (Python-only, ultra-fast)
Digunakan sebagai pre-filter sebelum memanggil AI untuk hemat token.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class ICTAnalyzer:
    def quick_check(self, market_context: dict) -> dict:
        h1 = market_context.get("h1", {})
        m5 = market_context.get("m5", {})
        h1_closed = h1.get("candles", [])[:-1] if h1.get("candles") else []
        m5_closed = m5.get("candles", [])[:-1] if m5.get("candles") else []

        h1_bos = self._detect_bos(h1.get("candles", []))
        h1_bos_type = h1_bos.get("direction", "") if h1_bos else ""

        return {
            "h1_bias": self._detect_bias(h1.get("candles", [])),
            "h1_fvg": self._find_fvg(h1_closed, bos_direction=h1_bos_type),
            "h1_bos": h1_bos,
            "m5_mss": self._detect_mss(m5.get("candles", [])),
            "m5_fvg": self._find_fvg(m5_closed),
            "liquidity_pools": self._identify_liquidity(h1.get("candles", [])),
        }

    def _detect_bias(self, candles: list) -> dict:
        if len(candles) < 10: return {"direction": "neutral", "reason": "insufficient data"}
        lookback = candles[-20:]
        swing_highs = []
        swing_lows = []
        for i in range(1, len(lookback) - 1):
            if lookback[i]["high"] > lookback[i-1]["high"] and lookback[i]["high"] > lookback[i+1]["high"]:
                swing_highs.append({"index": i, "price": lookback[i]["high"]})
            if lookback[i]["low"] < lookback[i-1]["low"] and lookback[i]["low"] < lookback[i+1]["low"]:
                swing_lows.append({"index": i, "price": lookback[i]["low"]})
        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {"direction": "neutral", "reason": "no clear swings"}
        last_2_highs = swing_highs[-2:]
        last_2_lows = swing_lows[-2:]
        hh = last_2_highs[1]["price"] > last_2_highs[0]["price"]
        hl = last_2_lows[1]["price"] > last_2_lows[0]["price"]
        lh = last_2_highs[1]["price"] < last_2_highs[0]["price"]
        ll = last_2_lows[1]["price"] < last_2_lows[0]["price"]
        if hh and hl:
            return {"direction": "bullish", "reason": "HH+HL", "last_sh": last_2_highs[-1]["price"], "last_sl": last_2_lows[-1]["price"]}
        elif lh and ll:
            return {"direction": "bearish", "reason": "LH+LL", "last_sh": last_2_highs[-1]["price"], "last_sl": last_2_lows[-1]["price"]}
        return {"direction": "neutral", "reason": "ranging", "last_sh": last_2_highs[-1]["price"], "last_sl": last_2_lows[-1]["price"]}

    def _find_fvg(self, candles: list, bos_direction: str = "") -> list:
        if len(candles) < 3: return []
        fvgs = []
        lookback = candles[-20:]
        for i in range(len(lookback) - 2):
            c1, c3 = lookback[i], lookback[i+2]
            if c1["high"] < c3["low"]:
                fvgs.append({"type": "bullish", "high": c3["low"], "low": c1["high"], "filled": False})
            if c1["low"] > c3["high"]:
                fvgs.append({"type": "bearish", "high": c1["low"], "low": c3["high"], "filled": False})
        if candles:
            last = candles[-1]["close"]
            valid = []
            for f in fvgs:
                if bos_direction and f["type"] != bos_direction: continue
                if f["type"] == "bullish" and last > f["low"]: f["filled"] = last >= f["high"]
                elif f["type"] == "bearish" and last < f["high"]: f["filled"] = last <= f["low"]
                if not f["filled"]: valid.append(f)
            return valid[-4:]
        return [f for f in fvgs if f["type"] == bos_direction][-4:] if bos_direction else fvgs[-4:]

    def _detect_bos(self, candles: list) -> Optional[dict]:
        if len(candles) < 15: return None
        lookback = candles[-15:]
        last = lookback[-1]
        recent = lookback[:-1]
        sh = max(c["high"] for c in recent[-10:])
        sl = min(c["low"] for c in recent[-10:])
        if last["close"] > sh: return {"type": "bullish_bos", "level": sh, "close": last["close"]}
        if last["close"] < sl: return {"type": "bearish_bos", "level": sl, "close": last["close"]}
        return None

    def _detect_mss(self, candles: list) -> Optional[dict]:
        if len(candles) < 10: return None
        lookback = candles[-10:]
        last, prev = lookback[-1], lookback[-2]
        if last["close"] > prev["high"] and last["open"] < prev["low"]: return {"type": "bullish_mss"}
        if last["close"] < prev["low"] and last["open"] > prev["high"]: return {"type": "bearish_mss"}
        return None

    def _identify_liquidity(self, candles: list) -> dict:
        if len(candles) < 20: return {}
        highs = [c["high"] for c in candles[-20:]]
        lows = [c["low"] for c in candles[-20:]]
        return {"recent_high": max(highs), "recent_low": min(lows)}