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

    @staticmethod
    def _price_decimals(price: float) -> int:
        """Tentukan jumlah desimal berdasarkan magnitude harga."""
        if price == 0:
            return 8
        if price >= 1000:   return 2   # BTC, ETH high
        if price >= 10:     return 3   # SOL, BNB, TAO
        if price >= 1:      return 4   # XRP, DOGE
        if price >= 0.1:    return 5   # FARTCOIN ~0.19
        if price >= 0.01:   return 6   # koin murah
        return 8                        # XVG, BONK, dll

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

            # Tentukan presisi dari candle pertama
            sample_price = h1_candles[0].get("close", 1.0) if h1_candles else 1.0
            dec = self._price_decimals(sample_price)

            # Format candle: hanya data yang AI butuhkan
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
            return {
                "symbol":  self.symbol,
                "price":   round(bid, self._price_decimals(bid)),
                "spread":  round(price.get("spread", 0), 8),
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
