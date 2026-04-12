"""
ICT Analyzer - Analisis struktur market ICT
Mengidentifikasi: Order Block, FVG, BOS, CHoCH, MSS
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class ICTAnalyzer:
    """
    Analisis ICT berbasis Python untuk validasi awal sebelum dikirim ke AI.
    Memberikan konteks terstruktur agar AI lebih akurat.
    """

    def quick_check(self, market_context: dict) -> dict:
        """Analisis cepat ICT dari data market yang sudah disiapkan."""
        m15 = market_context.get("m15", {})
        m1 = market_context.get("m1", {})

        result = {
            "m15_bias": self._detect_bias(m15.get("candles", [])),
            "m15_ob": self._find_order_blocks(m15.get("candles", [])),
            "m15_fvg": self._find_fvg(m15.get("candles", [])),
            "m15_bos": self._detect_bos(m15.get("candles", [])),
            "m1_mss": self._detect_mss(m1.get("candles", [])),
            "m1_fvg": self._find_fvg(m1.get("candles", [])),
            "liquidity_pools": self._identify_liquidity(m15.get("candles", [])),
        }

        logger.debug(f"ICT Analysis: {result}")
        return result

    def _detect_bias(self, candles: list) -> dict:
        """
        Tentukan bias M15 berdasarkan struktur swing.
        HH + HL = bullish, LH + LL = bearish
        """
        if len(candles) < 10:
            return {"direction": "neutral", "reason": "insufficient data"}

        # Ambil swing high/low dari 20 candle terakhir
        lookback = candles[-20:]
        highs = [c["high"] for c in lookback]
        lows = [c["low"] for c in lookback]

        # Simple swing detection (minimal 3 candle)
        swing_highs = []
        swing_lows = []

        for i in range(1, len(lookback) - 1):
            if lookback[i]["high"] > lookback[i - 1]["high"] and lookback[i]["high"] > lookback[i + 1]["high"]:
                swing_highs.append({"index": i, "price": lookback[i]["high"]})
            if lookback[i]["low"] < lookback[i - 1]["low"] and lookback[i]["low"] < lookback[i + 1]["low"]:
                swing_lows.append({"index": i, "price": lookback[i]["low"]})

        if len(swing_highs) < 2 or len(swing_lows) < 2:
            return {"direction": "neutral", "reason": "no clear swings"}

        # Check last 2 swing highs dan lows
        last_2_highs = swing_highs[-2:]
        last_2_lows = swing_lows[-2:]

        hh = last_2_highs[1]["price"] > last_2_highs[0]["price"]
        hl = last_2_lows[1]["price"] > last_2_lows[0]["price"]
        lh = last_2_highs[1]["price"] < last_2_highs[0]["price"]
        ll = last_2_lows[1]["price"] < last_2_lows[0]["price"]

        if hh and hl:
            return {
                "direction": "bullish",
                "reason": "HH + HL structure",
                "last_swing_high": last_2_highs[-1]["price"],
                "last_swing_low": last_2_lows[-1]["price"],
            }
        elif lh and ll:
            return {
                "direction": "bearish",
                "reason": "LH + LL structure",
                "last_swing_high": last_2_highs[-1]["price"],
                "last_swing_low": last_2_lows[-1]["price"],
            }
        else:
            return {
                "direction": "neutral",
                "reason": "mixed structure - ranging",
                "last_swing_high": last_2_highs[-1]["price"],
                "last_swing_low": last_2_lows[-1]["price"],
            }

    def _find_order_blocks(self, candles: list) -> list:
        """
        Cari Order Block:
        Bullish OB: last bearish candle sebelum impulse move bullish
        Bearish OB: last bullish candle sebelum impulse move bearish
        """
        if len(candles) < 5:
            return []

        order_blocks = []
        lookback = candles[-30:]

        for i in range(1, len(lookback) - 3):
            current = lookback[i]
            next_3 = lookback[i + 1: i + 4]

            # Ukur impulse: 3 candle ke depan
            impulse_range = max(c["high"] for c in next_3) - min(c["low"] for c in next_3)
            current_range = current["high"] - current["low"]

            # Bullish OB: bearish candle diikuti strong bullish move
            is_bearish = current["close"] < current["open"]
            strong_bullish_move = all(c["close"] > c["open"] for c in next_3[:2])
            if is_bearish and strong_bullish_move and impulse_range > current_range * 1.5:
                order_blocks.append({
                    "type": "bullish",
                    "high": current["high"],
                    "low": current["low"],
                    "open": current["open"],
                    "close": current["close"],
                    "timestamp": current.get("timestamp", ""),
                    "mitigation": False,
                })

            # Bearish OB: bullish candle diikuti strong bearish move
            is_bullish = current["close"] > current["open"]
            strong_bearish_move = all(c["close"] < c["open"] for c in next_3[:2])
            if is_bullish and strong_bearish_move and impulse_range > current_range * 1.5:
                order_blocks.append({
                    "type": "bearish",
                    "high": current["high"],
                    "low": current["low"],
                    "open": current["open"],
                    "close": current["close"],
                    "timestamp": current.get("timestamp", ""),
                    "mitigation": False,
                })

        # Return hanya yang belum dimitigasi (harga belum touch kembali)
        if candles:
            last_price = candles[-1]["close"]
            valid_obs = []
            for ob in order_blocks:
                if ob["type"] == "bullish" and last_price > ob["low"]:
                    # Belum ditembus ke bawah
                    if last_price <= ob["high"] * 1.005:  # Harga di area OB atau dekat
                        valid_obs.append(ob)
                    elif last_price > ob["high"]:
                        ob["mitigation"] = True
                        valid_obs.append(ob)
                elif ob["type"] == "bearish" and last_price < ob["high"]:
                    if last_price >= ob["low"] * 0.995:
                        valid_obs.append(ob)
                    elif last_price < ob["low"]:
                        ob["mitigation"] = True
                        valid_obs.append(ob)

            return valid_obs[-5:]  # Return max 5 OB terbaru

        return order_blocks[-5:]

    def _find_fvg(self, candles: list) -> list:
        """
        Fair Value Gap (Imbalance):
        Bullish FVG: candle[i].high < candle[i+2].low (gap antara high C1 dan low C3)
        Bearish FVG: candle[i].low > candle[i+2].high
        """
        if len(candles) < 3:
            return []

        fvgs = []
        lookback = candles[-20:]

        for i in range(len(lookback) - 2):
            c1 = lookback[i]
            c3 = lookback[i + 2]

            # Bullish FVG
            if c1["high"] < c3["low"]:
                gap_size = c3["low"] - c1["high"]
                fvgs.append({
                    "type": "bullish",
                    "high": c3["low"],
                    "low": c1["high"],
                    "gap_size": gap_size,
                    "midpoint": (c3["low"] + c1["high"]) / 2,
                    "timestamp": lookback[i + 1].get("timestamp", ""),
                    "filled": False,
                })

            # Bearish FVG
            if c1["low"] > c3["high"]:
                gap_size = c1["low"] - c3["high"]
                fvgs.append({
                    "type": "bearish",
                    "high": c1["low"],
                    "low": c3["high"],
                    "gap_size": gap_size,
                    "midpoint": (c1["low"] + c3["high"]) / 2,
                    "timestamp": lookback[i + 1].get("timestamp", ""),
                    "filled": False,
                })

        # Filter FVG yang belum terisi
        if candles:
            last_price = candles[-1]["close"]
            unfilled = []
            for fvg in fvgs:
                if fvg["type"] == "bullish" and last_price > fvg["low"]:
                    if last_price < fvg["high"]:
                        fvg["filled"] = False  # Partially atau belum terisi
                    else:
                        fvg["filled"] = True
                    unfilled.append(fvg)
                elif fvg["type"] == "bearish" and last_price < fvg["high"]:
                    if last_price > fvg["low"]:
                        fvg["filled"] = False
                    else:
                        fvg["filled"] = True
                    unfilled.append(fvg)
            return unfilled[-4:]

        return fvgs[-4:]

    def _detect_bos(self, candles: list) -> Optional[dict]:
        """
        Break of Structure (BOS) - konfirmasi kelanjutan trend.
        BOS bullish: close di atas swing high sebelumnya
        BOS bearish: close di bawah swing low sebelumnya
        """
        if len(candles) < 15:
            return None

        lookback = candles[-15:]
        last = lookback[-1]

        # Cari swing high/low dari 10 candle sebelum last
        recent = lookback[:-1]
        swing_high = max(c["high"] for c in recent[-10:])
        swing_low = min(c["low"] for c in recent[-10:])

        if last["close"] > swing_high:
            return {
                "type": "bullish_bos",
                "level": swing_high,
                "close": last["close"],
                "timestamp": last.get("timestamp", ""),
            }
        elif last["close"] < swing_low:
            return {
                "type": "bearish_bos",
                "level": swing_low,
                "close": last["close"],
                "timestamp": last.get("timestamp", ""),
            }
        return None

    def _detect_mss(self, candles: list) -> Optional[dict]:
        """
        Market Structure Shift (MSS) di M1.
        Reversal dari trend sebelumnya - sinyal entry potensial.
        """
        if len(candles) < 10:
            return None

        lookback = candles[-10:]

        # Deteksi engulfing candle sebagai MSS sederhana
        last = lookback[-1]
        prev = lookback[-2]

        # Bullish MSS: bullish engulfing (close > prev high)
        if last["close"] > prev["high"] and last["open"] < prev["low"]:
            return {
                "type": "bullish_mss",
                "engulfing_high": last["high"],
                "engulfing_low": last["low"],
                "timestamp": last.get("timestamp", ""),
            }

        # Bearish MSS: bearish engulfing
        if last["close"] < prev["low"] and last["open"] > prev["high"]:
            return {
                "type": "bearish_mss",
                "engulfing_high": last["high"],
                "engulfing_low": last["low"],
                "timestamp": last.get("timestamp", ""),
            }

        # Alternative MSS: BOS di M1
        m1_high = max(c["high"] for c in lookback[-5:-1])
        m1_low = min(c["low"] for c in lookback[-5:-1])

        if last["close"] > m1_high:
            return {
                "type": "bullish_mss_bos",
                "level": m1_high,
                "close": last["close"],
                "timestamp": last.get("timestamp", ""),
            }
        elif last["close"] < m1_low:
            return {
                "type": "bearish_mss_bos",
                "level": m1_low,
                "close": last["close"],
                "timestamp": last.get("timestamp", ""),
            }

        return None

    def _identify_liquidity(self, candles: list) -> dict:
        """
        Identifikasi liquidity pools:
        - Equal highs/lows (magnet untuk harga)
        - Previous day high/low
        - Round numbers
        """
        if len(candles) < 20:
            return {}

        lookback = candles[-20:]
        highs = [c["high"] for c in lookback]
        lows = [c["low"] for c in lookback]

        return {
            "recent_high": max(highs),
            "recent_low": min(lows),
            "prev_session_high": max(highs[-8:-4]),
            "prev_session_low": min(lows[-8:-4]),
        }

    # ─────────────────────────────────────────────────────
    # IDM (Inducement) Detection
    # Bullish IDM:
    #   Candle A buat swing high
    #   Minimal 1 candle "kosong" setelahnya (tidak tembus high A, close maupun wick)
    #   Candle selanjutnya tembus (high > high A)
    #   High candle A itulah level IDM yang harus disentuh untuk konfirmasi
    # ─────────────────────────────────────────────────────

    def find_idm(self, candles: list, direction: str = "bullish") -> list:
        """
        Cari semua IDM (Inducement) pada candle list.
        direction: 'bullish' atau 'bearish'
        Return: list of IDM dict, terbaru di belakang
        """
        idms = []
        if len(candles) < 4:
            return idms

        if direction == "bullish":
            # Cari swing high (candle A) yang jadi IDM bullish
            for i in range(1, len(candles) - 2):
                candle_a = candles[i]
                high_a = candle_a["high"]

                # Candle A harus lebih tinggi dari sebelum dan sesudahnya (swing high)
                if candle_a["high"] <= candles[i-1]["high"]:
                    continue

                # Cari minimal 1 candle kosong setelah A (tidak tembus high A sama sekali)
                gap_found = False
                tembus_idx = None

                for j in range(i+1, min(i+10, len(candles))):
                    c = candles[j]
                    if c["high"] >= high_a:
                        # Candle ini tembus — cek apakah ada gap sebelumnya
                        if gap_found:
                            tembus_idx = j
                        break
                    else:
                        # Candle ini tidak tembus — bisa jadi "kosong"
                        gap_found = True

                if tembus_idx and gap_found:
                    idms.append({
                        "type": "bullish_idm",
                        "level": round(high_a, 2),        # Harga IDM = high candle A (presisi)
                        "candle_a_idx": i,
                        "candle_a_time": candle_a.get("timestamp", ""),
                        "tembus_idx": tembus_idx,
                        "tembus_time": candles[tembus_idx].get("timestamp", ""),
                        "status": "active",               # active | touched
                        "candle_a_open": candle_a["open"],
                        "candle_a_close": candle_a["close"],
                        "candle_a_low": candle_a["low"],
                    })

        elif direction == "bearish":
            # IDM bearish: swing low → gap (tidak tembus low) → tembus low
            for i in range(1, len(candles) - 2):
                candle_a = candles[i]
                low_a = candle_a["low"]

                if candle_a["low"] >= candles[i-1]["low"]:
                    continue

                gap_found = False
                tembus_idx = None

                for j in range(i+1, min(i+10, len(candles))):
                    c = candles[j]
                    if c["low"] <= low_a:
                        if gap_found:
                            tembus_idx = j
                        break
                    else:
                        gap_found = True

                if tembus_idx and gap_found:
                    idms.append({
                        "type": "bearish_idm",
                        "level": round(low_a, 2),
                        "candle_a_idx": i,
                        "candle_a_time": candle_a.get("timestamp", ""),
                        "tembus_idx": tembus_idx,
                        "tembus_time": candles[tembus_idx].get("timestamp", ""),
                        "status": "active",
                        "candle_a_open": candle_a["open"],
                        "candle_a_close": candle_a["close"],
                        "candle_a_high": candle_a["high"],
                    })

        # Kembalikan IDM terbaru (berdasarkan candle_a_idx terbesar)
        return sorted(idms, key=lambda x: x["candle_a_idx"])

    def get_latest_idm(self, candles: list, direction: str = "bullish") -> dict | None:
        """Ambil IDM terbaru saja."""
        idms = self.find_idm(candles, direction)
        return idms[-1] if idms else None

    def msnr_level(self, candles: list, direction: str = "support") -> list:
        """
        MSNR (Malaysian Support/Resistance):
        Support = close candle bullish terendah di zona
        Resistance = close candle bearish tertinggi di zona
        Wick diabaikan — hanya close.
        """
        if len(candles) < 5:
            return []

        levels = []
        lookback = candles[-30:]

        if direction == "support":
            # Cari cluster close bullish di level rendah
            closes = [(i, c["close"]) for i, c in enumerate(lookback)
                      if c["close"] > c["open"]]  # candle bullish
            # Ambil yang paling rendah sebagai support
            closes.sort(key=lambda x: x[1])
            seen = set()
            for idx, price in closes[:5]:
                rounded = round(price, 1)
                if rounded not in seen:
                    seen.add(rounded)
                    levels.append({
                        "type": "msnr_support",
                        "level": round(price, 2),
                        "source": "close_bullish_candle",
                        "timestamp": lookback[idx].get("timestamp", ""),
                    })

        elif direction == "resistance":
            closes = [(i, c["close"]) for i, c in enumerate(lookback)
                      if c["close"] < c["open"]]  # candle bearish
            closes.sort(key=lambda x: x[1], reverse=True)
            seen = set()
            for idx, price in closes[:5]:
                rounded = round(price, 1)
                if rounded not in seen:
                    seen.add(rounded)
                    levels.append({
                        "type": "msnr_resistance",
                        "level": round(price, 2),
                        "source": "close_bearish_candle",
                        "timestamp": lookback[idx].get("timestamp", ""),
                    })

        return levels

    def check_bos_m1(self, candles: list, direction: str = "bullish", idm_level: float = 0) -> dict | None:
        """
        Cek BOS di M1 setelah IDM tersentuh.
        BOS bullish: setelah IDM bullish disentuh, harga close di atas swing high terakhir
        BOS bearish: setelah IDM bearish disentuh, harga close di bawah swing low terakhir
        """
        if len(candles) < 5 or idm_level == 0:
            return None

        # Cari swing high/low setelah IDM
        recent = candles[-15:]

        if direction == "bullish":
            swing_highs = []
            for i in range(1, len(recent)-1):
                if recent[i]["high"] > recent[i-1]["high"] and recent[i]["high"] > recent[i+1]["high"]:
                    swing_highs.append(recent[i]["high"])

            if not swing_highs:
                return None

            last_sh = max(swing_highs)
            last_candle = recent[-1]

            if last_candle["close"] > last_sh:
                return {
                    "type": "bullish_bos_m1",
                    "level": round(last_sh, 2),
                    "close": round(last_candle["close"], 2),
                    "timestamp": last_candle.get("timestamp", ""),
                }

        elif direction == "bearish":
            swing_lows = []
            for i in range(1, len(recent)-1):
                if recent[i]["low"] < recent[i-1]["low"] and recent[i]["low"] < recent[i+1]["low"]:
                    swing_lows.append(recent[i]["low"])

            if not swing_lows:
                return None

            last_sl = min(swing_lows)
            last_candle = recent[-1]

            if last_candle["close"] < last_sl:
                return {
                    "type": "bearish_bos_m1",
                    "level": round(last_sl, 2),
                    "close": round(last_candle["close"], 2),
                    "timestamp": last_candle.get("timestamp", ""),
                }

        return None
