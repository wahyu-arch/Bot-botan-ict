"""
DataProvider — Satu-satunya tugas Python selain eksekusi order:
Ambil data mentah candle dari market dan format untuk AI.
AI yang memutuskan semua analisis.
"""
import os, logging
from market_data import MarketDataFetcher

logger = logging.getLogger(__name__)


class DataProvider:
    """Kasih data mentah ke AI. Tidak ada analisis di sini."""

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.fetcher = MarketDataFetcher(symbol)

    def get_raw(self) -> dict | None:
        """Ambil candle H1 dan M5 mentah. Return dict atau None kalau gagal."""
        try:
            ctx = self.fetcher.get_full_context()
            if not ctx:
                return None

            h1_candles = ctx.get("h1", {}).get("candles", [])
            m5_candles = ctx.get("m5", {}).get("candles", [])
            price      = ctx.get("current_price", {})

            if not h1_candles or not m5_candles:
                return None

            # Format candle: hanya data yang AI butuhkan
            def fmt(candles, skip_last=True):
                data = candles[:-1] if skip_last and len(candles) > 1 else candles
                return [
                    {
                        "i":  i,
                        "o":  round(c["open"],  2),
                        "h":  round(c["high"],  2),
                        "l":  round(c["low"],   2),
                        "c":  round(c["close"], 2),
                        "ts": c.get("timestamp", "")[:16],
                        "bull": c["close"] > c["open"],
                    }
                    for i, c in enumerate(data)
                ]

            return {
                "symbol":  self.symbol,
                "price":   round(price.get("bid", 0), 2),
                "spread":  round(price.get("spread", 0), 4),
                "h1":      fmt(h1_candles),   # closed H1 candles
                "m5":      fmt(m5_candles),   # closed M5 candles
                "h1_live": fmt(h1_candles, skip_last=False)[-1],  # candle H1 berjalan
                "m5_live": fmt(m5_candles, skip_last=False)[-1],  # candle M5 berjalan
            }
        except Exception as e:
            logger.error(f"[DATA] Gagal ambil data: {e}")
            return None

    def format_candles_for_ai(self, candles: list, limit: int = 50) -> str:
        """Format candle jadi teks ringkas untuk prompt AI."""
        recent = candles[-limit:]
        lines = []
        for c in recent:
            arrow = "↑" if c.get("bull") else "↓"
            lines.append(
                f"[{c['i']:3d}] {c['ts']} {arrow} O:{c['o']} H:{c['h']} L:{c['l']} C:{c['c']}"
            )
        return "\n".join(lines)
