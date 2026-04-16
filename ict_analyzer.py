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
        h1 = market_context.get("h1", {})
        m5 = market_context.get("m5", {})

        # H1: hanya struktur BOS + FVG (tidak ada OB)
        # FVG hanya dari candle yang sudah close (skip candle terakhir yg masih berjalan)
        h1_closed = h1.get("candles", [])[:-1] if h1.get("candles") else []
        m5_closed = m5.get("candles", [])[:-1] if m5.get("candles") else []

        # Deteksi BOS H1 dulu supaya FVG bisa difilter sesuai arah BOS
        h1_bos = self._detect_bos(h1.get("candles", []))
        h1_bos_type = h1_bos.get("direction", "") if h1_bos else ""  # "bullish" atau "bearish"

        result = {
            "h1_bias": self._detect_bias(h1.get("candles", [])),
            "h1_fvg": self._find_fvg(h1_closed, bos_direction=h1_bos_type),
            "h1_bos": h1_bos,
            "m5_mss": self._detect_mss(m5.get("candles", [])),
            "m5_fvg": self._find_fvg(m5_closed),  # M5 FVG tidak difilter arah
            "liquidity_pools": self._identify_liquidity(h1.get("candles", [])),
        }

        logger.debug(f"ICT Analysis: {result}")
        return result

    def _detect_bias(self, candles: list) -> dict:
        """
        Tentukan bias H1 berdasarkan struktur swing.
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

    def _find_fvg(self, candles: list, bos_direction: str = "") -> list:
        """
        Fair Value Gap (Imbalance):
        Bullish FVG: candle[i].high < candle[i+2].low (gap antara high C1 dan low C3)
        Bearish FVG: candle[i].low > candle[i+2].high

        bos_direction: "bullish" → hanya return FVG bullish (retrace ke bawah ke FVG)
                       "bearish" → hanya return FVG bearish
                       ""        → return semua (untuk M5, dll)
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

        # Filter FVG yang belum terisi + filter arah sesuai BOS
        if candles:
            last_price = candles[-1]["close"]
            unfilled = []
            for fvg in fvgs:
                # Filter arah: BOS bullish = hanya FVG bullish, BOS bearish = hanya FVG bearish
                if bos_direction == "bullish" and fvg["type"] != "bullish":
                    continue
                if bos_direction == "bearish" and fvg["type"] != "bearish":
                    continue

                if fvg["type"] == "bullish" and last_price > fvg["low"]:
                    fvg["filled"] = last_price >= fvg["high"]
                    unfilled.append(fvg)
                elif fvg["type"] == "bearish" and last_price < fvg["high"]:
                    fvg["filled"] = last_price <= fvg["low"]
                    unfilled.append(fvg)
            return unfilled[-4:]

        # Filter arah jika tidak ada last_price
        if bos_direction:
            fvgs = [f for f in fvgs if f["type"] == bos_direction]
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
        Market Structure Shift (MSS) di M5.
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

        # Alternative MSS: BOS di M5
        m5_high = max(c["high"] for c in lookback[-5:-1])
        m5_low = min(c["low"] for c in lookback[-5:-1])

        if last["close"] > m5_high:
            return {
                "type": "bullish_mss_bos",
                "level": m5_high,
                "close": last["close"],
                "timestamp": last.get("timestamp", ""),
            }
        elif last["close"] < m5_low:
            return {
                "type": "bearish_mss_bos",
                "level": m5_low,
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

    def find_idm(self, candles: list, direction: str = "bullish",
                 gap_min: int = 1, max_search: int = 10) -> list:
        """
        Cari semua IDM (Inducement) pada candle list.
        direction: 'bullish' atau 'bearish'
        gap_min: minimum candle kosong (dari rules)
        max_search: max candle yang dicari setelah A (dari rules)
        """
        idms = []
        if len(candles) < 4:
            return idms

        if direction == "bullish":
            for i in range(1, len(candles) - 2):
                candle_a = candles[i]
                high_a = candle_a["high"]

                if candle_a["high"] <= candles[i-1]["high"]:
                    continue

                gap_count = 0
                tembus_idx = None

                for j in range(i+1, min(i+max_search, len(candles))):
                    c = candles[j]
                    if c["high"] >= high_a:
                        if gap_count >= gap_min:
                            tembus_idx = j
                        break
                    else:
                        gap_count += 1

                if tembus_idx and gap_count >= gap_min:
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

                gap_count = 0
                tembus_idx = None

                for j in range(i+1, min(i+max_search, len(candles))):
                    c = candles[j]
                    if c["low"] <= low_a:
                        if gap_count >= gap_min:
                            tembus_idx = j
                        break
                    else:
                        gap_count += 1

                if tembus_idx and gap_count >= gap_min:
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

    def get_latest_idm(self, candles: list, direction: str = "bullish",
                         gap_min: int = 1, max_search: int = 10) -> dict | None:
        """Ambil IDM terbaru saja."""
        idms = self.find_idm(candles, direction, gap_min=gap_min, max_search=max_search)
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

    def check_bos_m5(self, candles: list, direction: str = "bullish", idm_level: float = 0) -> dict | None:
        """
        Cek BOS di M5 setelah IDM tersentuh.
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
                    "type": "bullish_bos_m5",
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
                    "type": "bearish_bos_m5",
                    "level": round(last_sl, 2),
                    "close": round(last_candle["close"], 2),
                    "timestamp": last_candle.get("timestamp", ""),
                }

        return None

    # ─────────────────────────────────────────────────────
    # AI-1 Logic: BOS H1 → IDM H1 → Swing High/Low
    # ─────────────────────────────────────────────────────

    def find_bos_h1(self, candles: list, lookback: int = 40, swing_min: int = 1,
                    swing_left: int = 8, swing_right: int = 8) -> dict | None:
        """
        Deteksi BOS H1 terbaru.
        BOS bullish: close candle melewati swing high sebelumnya
        BOS bearish: close candle melewati swing low sebelumnya
        Return dict dengan type, level, bos_candle_idx, dan swing_high/low sebelum BOS.
        """
        if len(candles) < 10:
            return None

        lookback = candles[-lookback:]
        n = len(lookback)

        # Kumpulkan swing highs dan lows
        swing_highs = []  # (idx, price)
        swing_lows  = []  # (idx, price)
        # Pakai swing_left dan swing_right dari logic rules
        sl_r = min(swing_left, swing_right, 3)  # min 3 untuk stabilitas
        for i in range(sl_r, n - sl_r):
            left_high  = all(lookback[i]["high"] >= lookback[j]["high"] for j in range(max(0,i-swing_left), i))
            right_high = all(lookback[i]["high"] >= lookback[j]["high"] for j in range(i+1, min(n,i+swing_right+1)))
            left_low   = all(lookback[i]["low"] <= lookback[j]["low"]  for j in range(max(0,i-swing_left), i))
            right_low  = all(lookback[i]["low"] <= lookback[j]["low"]  for j in range(i+1, min(n,i+swing_right+1)))
            if left_high and right_high:
                swing_highs.append((i, lookback[i]["high"]))
            if left_low and right_low:
                swing_lows.append((i, lookback[i]["low"]))

        if not swing_highs or not swing_lows:
            return None

        # Cari BOS dari candle terbaru ke belakang
        for i in range(n-1, 5, -1):
            c = lookback[i]

            # BOS bullish: close di atas swing high sebelumnya
            prev_highs = [(idx, p) for idx, p in swing_highs if idx < i]
            if prev_highs:
                last_sh_idx, last_sh = prev_highs[-1]
                if c["close"] > last_sh:
                    # Cari swing low sebelum BOS (untuk IDM H1 bullish)
                    prev_lows = [(idx, p) for idx, p in swing_lows if idx < last_sh_idx]
                    last_sl = prev_lows[-1][1] if prev_lows else lookback[0]["low"]

                    # Cari swing high sejak BOS (high tertinggi setelah BOS terbentuk)
                    post_bos_highs = [lookback[j]["high"] for j in range(last_sh_idx, i+1)]
                    sh_since_bos = max(post_bos_highs) if post_bos_highs else last_sh

                    return {
                        "type": "bullish_bos",
                        "level": round(last_sh, 2),         # level yang ditembus
                        "bos_close": round(c["close"], 2),
                        "bos_candle_idx": i,
                        "bos_time": c.get("timestamp", ""),
                        "sh_since_bos": round(sh_since_bos, 2),  # SH tertinggi sejak BOS
                        "sl_before_bos": round(last_sl, 2),      # SL sebelum BOS
                        # IDM H1 yang harus disentuh = IDM bullish H1 (high candle A)
                        # dicari terpisah via find_idm_after_bos()
                    }

            # BOS bearish: close di bawah swing low sebelumnya
            prev_lows = [(idx, p) for idx, p in swing_lows if idx < i]
            if prev_lows:
                last_sl_idx, last_sl = prev_lows[-1]
                if c["close"] < last_sl:
                    prev_highs_before = [(idx, p) for idx, p in swing_highs if idx < last_sl_idx]
                    last_sh = prev_highs_before[-1][1] if prev_highs_before else lookback[0]["high"]

                    post_bos_lows = [lookback[j]["low"] for j in range(last_sl_idx, i+1)]
                    sl_since_bos = min(post_bos_lows) if post_bos_lows else last_sl

                    return {
                        "type": "bearish_bos",
                        "level": round(last_sl, 2),
                        "bos_close": round(c["close"], 2),
                        "bos_candle_idx": i,
                        "bos_time": c.get("timestamp", ""),
                        "sl_since_bos": round(sl_since_bos, 2),  # SL terendah sejak BOS
                        "sh_before_bos": round(last_sh, 2),
                    }

        return None

    def find_idm_after_bos(self, candles: list, bos: dict,
                            gap_min: int = 1, max_search: int = 10) -> dict | None:
        """
        Cari IDM H1 yang harus disentuh SETELAH BOS terbentuk (untuk retrace).

        Logika:
        - BOS bullish H1 sudah terjadi → harga akan retrace turun
        - Untuk retrace, harga harus menyentuh IDM bullish H1 yang ada
        - IDM bullish = candle A buat swing high → gap (tidak tembus) → tembus
        - Yang dipantau = LOW candle A (bukan high) — karena saat retrace,
          harga akan turun dan low candle A adalah level IDM yang harus disentuh

        Dengan kata lain:
        - BOS bullish → cari IDM bullish H1 SETELAH BOS → pantau LOW candle A
        - BOS bearish → cari IDM bearish H1 SETELAH BOS → pantau HIGH candle A
        """
        if not bos:
            return None

        bos_type = bos.get("type", "")
        bos_idx  = bos.get("bos_candle_idx", 0)
        lookback = candles[-40:]

        # Candle setelah BOS terbentuk
        post_bos = lookback[bos_idx:]
        if len(post_bos) < 3:
            return None

        if bos_type == "bullish_bos":
            # Cari IDM bullish setelah BOS (swing high → gap → tembus high)
            idms = self.find_idm(post_bos, direction="bullish", gap_min=gap_min, max_search=max_search)
            if idms:
                idm = idms[-1]
                # Yang dipantau: LOW candle A (bukan high)
                # Saat retrace dari bullish, harga turun ke low candle A
                watch_level = idm.get("candle_a_low", idm["level"])
                idm["watch_level"] = round(watch_level, 2)  # low candle A
                idm["watch_condition"] = "touch"
                idm["context"] = "h1_bullish_bos_retrace"
                idm["description"] = (
                    f"IDM bullish H1: low candle A @ {watch_level:.2f} "
                    f"(high={idm['level']:.2f}) — retrace harus sentuh low ini"
                )
                return idm

        elif bos_type == "bearish_bos":
            # IDM bearish setelah BOS → pantau HIGH candle A
            idms = self.find_idm(post_bos, direction="bearish", gap_min=gap_min, max_search=max_search)
            if idms:
                idm = idms[-1]
                watch_level = idm.get("candle_a_high", idm["level"])
                idm["watch_level"] = round(watch_level, 2)  # high candle A
                idm["watch_condition"] = "touch"
                idm["context"] = "h1_bearish_bos_retrace"
                idm["description"] = (
                    f"IDM bearish H1: high candle A @ {watch_level:.2f} "
                    f"(low={idm['level']:.2f}) — retrace harus sentuh high ini"
                )
                return idm

        return None

    def check_idm_touched(self, candles: list, idm_level: float, direction: str) -> dict | None:
        """
        Cek apakah IDM H1 sudah disentuh (wick atau body boleh).
        direction: arah IDM ('bullish' → level adalah high, sentuh dari bawah ke atas lalu turun)

        Return: dict info candle yang menyentuh, atau None.
        """
        if not candles or idm_level <= 0:
            return None

        for c in reversed(candles[-20:]):
            # Sentuh = wick atau body melewati/menyentuh level
            if direction == "bullish":
                # Harga harus naik ke level IDM (wick high atau close menyentuh)
                if c["high"] >= idm_level:
                    return {
                        "touched": True,
                        "touch_candle_high": round(c["high"], 2),
                        "touch_candle_low": round(c["low"], 2),
                        "touch_candle_close": round(c["close"], 2),
                        "touch_time": c.get("timestamp", ""),
                        "touch_type": "body" if c["close"] >= idm_level else "wick",
                    }
            elif direction == "bearish":
                if c["low"] <= idm_level:
                    return {
                        "touched": True,
                        "touch_candle_high": round(c["high"], 2),
                        "touch_candle_low": round(c["low"], 2),
                        "touch_candle_close": round(c["close"], 2),
                        "touch_time": c.get("timestamp", ""),
                        "touch_type": "body" if c["close"] <= idm_level else "wick",
                    }
        return None

    def get_swing_range_after_idm_touch(self, candles: list, bos: dict, touch_info: dict) -> dict:
        """
        Setelah IDM H1 disentuh, hitung range untuk AI-2 cari IDM di M5:

        BOS bullish:
          SH = high tertinggi sejak BOS (sh_since_bos) — batas atas range
          SL = swing low SEBELUM BOS terbentuk (sl_before_bos) — batas bawah range
               Ini adalah HL yang membentuk struktur bullish, biasanya di area OB atau di bawahnya.
               Bukan low candle yang sentuh IDM — itu terlalu tinggi.

        BOS bearish:
          SH = swing high sebelum BOS (sh_before_bos)
          SL = low tertinggi sejak BOS (sl_since_bos)
        """
        bos_type = bos.get("type", "")

        if bos_type == "bullish_bos":
            sh = bos.get("sh_since_bos", 0)       # High tertinggi sejak BOS
            sl = bos.get("sl_before_bos", 0)       # Swing low SEBELUM BOS = HL struktur bullish
            # Fallback: kalau sl_before_bos tidak ada, gunakan touch candle low
            if sl == 0:
                sl = touch_info.get("touch_candle_low", 0)
            return {
                "swing_high": round(sh, 2),
                "swing_low": round(sl, 2),
                "m5_idm_direction": "bearish",
                "range_valid": sh > sl > 0,
            }
        elif bos_type == "bearish_bos":
            sh = bos.get("sh_before_bos", 0)       # Swing high sebelum BOS = LH struktur bearish
            sl = bos.get("sl_since_bos", 0)         # Low terendah sejak BOS
            if sh == 0:
                sh = touch_info.get("touch_candle_high", 0)
            return {
                "swing_high": round(sh, 2),
                "swing_low": round(sl, 2),
                "m5_idm_direction": "bullish",
                "range_valid": sh > sl > 0,
            }
        return {"swing_high": 0, "swing_low": 0, "m5_idm_direction": "unknown", "range_valid": False}

    def check_ob_sweep_fakeout(self, candles: list, obs: list, bos_type: str) -> dict:
        """
        Cek apakah ada OB sweep yang berpotensi fakeout.

        BOS bullish: OB di bawah harga saat ini ke-sweep (wick tembus low OB)
          → Fakeout kalau TIDAK ada close H1 di bawah OB low
          → Konfirmasi bearish kalau ada close H1 di bawah OB low

        BOS bearish: OB di atas ke-sweep
          → Fakeout kalau tidak ada close H1 di atas OB high
        """
        if not candles or not obs:
            return {"sweep_detected": False}

        last_candle = candles[-1]
        result = {"sweep_detected": False, "fakeout": False, "confirmed_break": False}

        for ob in obs:
            if bos_type == "bullish_bos":
                ob_low = ob.get("low", 0)
                ob_high = ob.get("high", 0)
                # Wick melewati OB low tapi close masih di atas
                if last_candle["low"] < ob_low and last_candle["close"] > ob_low:
                    result["sweep_detected"] = True
                    result["fakeout"] = True
                    result["ob_level"] = round(ob_low, 2)
                    result["message"] = f"OB sweep fakeout: wick ke {last_candle['low']:.2f} tapi close {last_candle['close']:.2f} masih di atas OB low {ob_low:.2f} — tunggu dulu"
                    break
                # Close di bawah OB low = konfirmasi bearish
                elif last_candle["close"] < ob_low:
                    result["sweep_detected"] = True
                    result["confirmed_break"] = True
                    result["ob_level"] = round(ob_low, 2)
                    result["message"] = f"OB break terkonfirmasi: close {last_candle['close']:.2f} di bawah OB low {ob_low:.2f}"
                    break

            elif bos_type == "bearish_bos":
                ob_high = ob.get("high", 0)
                if last_candle["high"] > ob_high and last_candle["close"] < ob_high:
                    result["sweep_detected"] = True
                    result["fakeout"] = True
                    result["ob_level"] = round(ob_high, 2)
                    result["message"] = f"OB sweep fakeout: wick ke {last_candle['high']:.2f} tapi close {last_candle['close']:.2f} masih di bawah OB high {ob_high:.2f} — tunggu dulu"
                    break
                elif last_candle["close"] > ob_high:
                    result["sweep_detected"] = True
                    result["confirmed_break"] = True
                    result["ob_level"] = round(ob_high, 2)
                    result["message"] = f"OB break terkonfirmasi: close {last_candle['close']:.2f} di atas OB high {ob_high:.2f}"
                    break

        return result

    def find_idm_in_range(self, candles: list, direction: str,
                          swing_high: float, swing_low: float,
                          gap_min: int = 1, max_search: int = 10) -> dict | None:
        """
        Cari IDM di M5 dalam range SH-SL yang diberikan AI-1.
        direction: 'bearish' (untuk BOS bullish H1) atau 'bullish' (untuk BOS bearish H1)

        Filter: IDM harus berada di dalam range swing_low < level < swing_high
        """
        if swing_high <= swing_low or not candles:
            return None

        # Filter candle yang berada dalam range harga
        relevant = [c for c in candles if swing_low <= c["low"] and c["high"] <= swing_high * 1.002]
        if len(relevant) < 4:
            relevant = candles  # fallback: gunakan semua jika tidak cukup

        idms = self.find_idm(relevant, direction=direction, gap_min=gap_min, max_search=max_search)

        # Filter: level IDM harus dalam range
        valid = [idm for idm in idms
                 if swing_low <= idm["level"] <= swing_high]

        return valid[-1] if valid else None

    def check_fvg_wick_touched(self, candles: list, bos_type: str) -> dict | None:
        """
        Cek apakah ada candle (closed) yang wick-menyentuh FVG H1.

        Wick touch = wick masuk zona FVG tapi body CLOSE masih di luar zona.
        Body close menembus FVG = FVG invalid (skip, cari FVG lain).

        BOS bullish → cari FVG bullish yang disentuh wick dari bawah (harga retrace turun)
        BOS bearish → cari FVG bearish yang disentuh wick dari atas (harga retrace naik)

        Return: {fvg, touch_candle, sh, sl} atau None
        """
        if len(candles) < 4:
            return None

        # Skip candle terakhir (masih berjalan)
        closed = candles[:-1]
        fvgs = self._find_fvg(closed)

        # Ambil FVG yang relevan dengan BOS
        if bos_type == "bullish_bos":
            relevant_fvgs = [f for f in fvgs if f["type"] == "bullish" and not f["filled"]]
        else:
            relevant_fvgs = [f for f in fvgs if f["type"] == "bearish" and not f["filled"]]

        if not relevant_fvgs:
            return None

        # Cek dari candle terbaru ke belakang (max 10 candle terakhir)
        for candle in reversed(closed[-10:]):
            c_high  = candle["high"]
            c_low   = candle["low"]
            c_close = candle["close"]
            c_open  = candle["open"]
            body_top = max(c_close, c_open)
            body_bot = min(c_close, c_open)

            for fvg in relevant_fvgs:
                fvg_low  = fvg["low"]
                fvg_high = fvg["high"]

                if bos_type == "bullish_bos":
                    # Retrace turun: wick bawah masuk FVG, tapi body close masih di atas fvg_low
                    wick_inside = c_low <= fvg_high  # wick menyentuh atau masuk zona
                    body_outside = body_bot >= fvg_low  # body close masih di atas batas bawah FVG
                    body_break = c_close < fvg_low  # body close tembus ke bawah = FVG invalid

                    if body_break:
                        fvg["filled"] = True  # mark invalid
                        continue
                    if wick_inside and body_outside:
                        return {
                            "fvg": fvg,
                            "touch_candle": candle,
                            "touch_type": "wick",
                            "touch_price": round(c_low, 2),
                        }

                else:  # bearish_bos
                    # Retrace naik: wick atas masuk FVG, body close masih di bawah fvg_high
                    wick_inside = c_high >= fvg_low
                    body_outside = body_top <= fvg_high
                    body_break = c_close > fvg_high

                    if body_break:
                        fvg["filled"] = True
                        continue
                    if wick_inside and body_outside:
                        return {
                            "fvg": fvg,
                            "touch_candle": candle,
                            "touch_type": "wick",
                            "touch_price": round(c_high, 2),
                        }
        return None

    def get_sh_sl_from_bos(self, candles: list, bos: dict) -> dict:
        """
        Ambil SH dan SL dari struktur BOS H1 untuk diserahkan ke AI-2.

        BOS bullish:
          SH = high tertinggi sejak BOS terbentuk (sebelum retrace)
          SL = swing low sebelum BOS (sl_before_bos) — level HL struktur bullish

        BOS bearish:
          SH = swing high sebelum BOS (sh_before_bos)
          SL = low terendah sejak BOS
        """
        if not bos:
            return {"swing_high": 0, "swing_low": 0, "m5_idm_direction": "unknown", "range_valid": False}

        bos_type = bos.get("type", "")

        if bos_type == "bullish_bos":
            sh = bos.get("sh_since_bos", 0)
            sl = bos.get("sl_before_bos", 0)
            return {
                "swing_high": round(sh, 2),
                "swing_low": round(sl, 2),
                "m5_idm_direction": "bearish",
                "range_valid": sh > sl > 0,
            }
        elif bos_type == "bearish_bos":
            sh = bos.get("sh_before_bos", 0)
            sl = bos.get("sl_since_bos", 0)
            return {
                "swing_high": round(sh, 2),
                "swing_low": round(sl, 2),
                "m5_idm_direction": "bullish",
                "range_valid": sh > sl > 0,
            }
        return {"swing_high": 0, "swing_low": 0, "m5_idm_direction": "unknown", "range_valid": False}
