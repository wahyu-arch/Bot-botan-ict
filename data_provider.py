"""
DataProvider — Ambil data candle dari market dan format untuk AI.
Masing-masing AI dapat jumlah candle yang berbeda sesuai kebutuhannya.
"""
import os, logging, time
from market_data import MarketDataFetcher

logger = logging.getLogger(__name__)

# Jumlah candle per AI sesuai rekomendasi Katyusha
CANDLE_LIMITS = {
    "hiura":   {"h1": 150, "m5": 0},    # Hiura butuh 150 H1 untuk swing detection
    "senanan": {"h1": 0,   "m5": 200},  # Senanan butuh 200 M5 untuk IDM hunt
    "shina":   {"h1": 0,   "m5": 200},  # Shina butuh M5 dari IDM start sampai sekarang
    "yusuf":   {"h1": 0,   "m5": 50},   # Yusuf cukup 50 M5 terakhir
    "default": {"h1": 150, "m5": 200},  # default: ambil maksimum
}

MAX_CANDLE_DELAY_SECONDS = 120  # candle dianggap stale kalau > 2 menit dari sekarang


class DataProvider:
    def __init__(self, symbol: str):
        self.symbol = symbol
        self.fetcher = MarketDataFetcher(symbol)
        self._last_fetch_ts: float = 0.0
        self._cached_raw: dict | None = None

    @staticmethod
    def _price_decimals(price: float) -> int:
        if price == 0:    return 8
        if price >= 1000: return 2
        if price >= 10:   return 3
        if price >= 1:    return 4
        if price >= 0.1:  return 5
        if price >= 0.01: return 6
        return 8

    def is_candle_fresh(self) -> bool:
        """Cek apakah data candle masih fresh (tidak stale)."""
        if self._last_fetch_ts == 0:
            return False
        age = time.time() - self._last_fetch_ts
        if age > MAX_CANDLE_DELAY_SECONDS:
            logger.warning(f"[DATA] Candle stale! Age: {age:.0f}s > {MAX_CANDLE_DELAY_SECONDS}s")
            return False
        return True

    def get_raw(self) -> dict | None:
        """Ambil candle H1 dan M5. Return dict atau None kalau gagal."""
        try:
            ctx = self.fetcher.get_full_context()
            if not ctx:
                return None

            h1_candles = ctx.get("h1", {}).get("candles", [])
            m5_candles = ctx.get("m5", {}).get("candles", [])
            price      = ctx.get("current_price", {})

            if not h1_candles or not m5_candles:
                return None

            # Cek minimum candle yang dibutuhkan Hiura
            if len(h1_candles) < 50:
                logger.warning(f"[DATA] H1 candle terlalu sedikit: {len(h1_candles)} < 50")
                return None

            sample_price = h1_candles[0].get("close", 1.0) if h1_candles else 1.0
            dec = self._price_decimals(sample_price)

            def fmt(candles, skip_last=True):
                data = candles[:-1] if skip_last and len(candles) > 1 else candles
                return [
                    {
                        "i":  i,
                        "o":  round(c["open"],  dec),
                        "h":  round(c["high"],  dec),
                        "l":  round(c["low"],   dec),
                        "c":  round(c["close"], dec),
                        "ts": c.get("timestamp", "")[:16],
                        "bull": c["close"] > c["open"],
                    }
                    for i, c in enumerate(data)
                ]

            bid = price.get("bid", 0)
            self._last_fetch_ts = time.time()
            raw = {
                "symbol":    self.symbol,
                "price":     round(bid, self._price_decimals(bid)),
                "spread":    round(price.get("spread", 0), 8),
                "h1":        fmt(h1_candles),
                "m5":        fmt(m5_candles),
                "h1_live":   fmt(h1_candles, skip_last=False)[-1],
                "m5_live":   fmt(m5_candles, skip_last=False)[-1],
                "fetched_at": self._last_fetch_ts,
            }
            self._cached_raw = raw
            return raw
        except Exception as e:
            logger.error(f"[DATA] Gagal ambil data: {e}")
            return None

    def get_candles_for_ai(self, ai_name: str, raw: dict,
                           idm_start_idx: int = 0) -> dict:
        """
        Return candle yang tepat untuk AI tertentu.
        Shina mendapat candle M5 dari IDM start + 50 candle sebelumnya.
        """
        limits = CANDLE_LIMITS.get(ai_name, CANDLE_LIMITS["default"])
        result = {}

        if limits["h1"] > 0:
            result["h1"] = raw.get("h1", [])[-limits["h1"]:]

        if limits["m5"] > 0:
            m5 = raw.get("m5", [])
            if ai_name == "shina" and idm_start_idx > 0:
                # Shina: dari 50 candle sebelum IDM start sampai sekarang
                start = max(0, idm_start_idx - 50)
                result["m5"] = m5[start:]
            else:
                result["m5"] = m5[-limits["m5"]:]

        return result

    def format_candles_for_ai(self, candles: list, limit: int = 50) -> str:
        """Format candle jadi teks tabel untuk prompt AI."""
        recent = candles[-limit:]
        lines = []
        for c in recent:
            arrow = "↑" if c.get("bull") else "↓"
            lines.append(
                f"[{c['i']:3d}] {c['ts']} {arrow} O:{c['o']} H:{c['h']} L:{c['l']} C:{c['c']}"
            )
        return "\n".join(lines)
