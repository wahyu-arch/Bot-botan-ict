"""
Market Data Fetcher - Bybit API (klines) + Mock
Mendukung: Bybit Linear Perp, atau Mock untuk testing
"""

import os
import logging
import random
from datetime import datetime, timezone, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


class MarketDataFetcher:
    """
    Fetcher candle data dari Bybit atau mock.
    Set DATA_SOURCE: "bybit" | "mock"
    """

    def __init__(self, symbol: str):
        self.symbol = symbol
        self.source = os.getenv("DATA_SOURCE", "mock").lower()
        self._init_mock_price()

        if self.source == "bybit":
            self._init_bybit()

    def _init_mock_price(self):
        defaults = {
            "BTCUSDT":      65000.0,
            "ETHUSDT":      3500.0,
            "SOLUSDT":      180.0,
            "XRPUSDT":      0.62,
            "BNBUSDT":      580.0,
            "DOGEUSDT":     0.18,
            "XVGUSDT":      0.0045,
            "FARTCOINUSDT": 0.95,
            "1000BONKUSDT": 0.028,
            "TAOUSDT":      240.0,
        }
        self.mock_price = defaults.get(self.symbol, 1.0)

    def _init_bybit(self):
        try:
            from pybit.unified_trading import HTTP
            testnet = os.getenv("BYBIT_TESTNET", "true").lower() == "true"
            # Klines tidak perlu auth, tapi pakai key jika tersedia
            api_key = os.getenv("BYBIT_API_KEY", "")
            api_secret = os.getenv("BYBIT_API_SECRET", "")
            self.bybit = HTTP(
                testnet=testnet,
                api_key=api_key if api_key else None,
                api_secret=api_secret if api_secret else None,
            )
            logger.info(f"Bybit market data initialized | testnet={testnet}")
        except ImportError:
            logger.warning("pybit tidak ada, fallback ke mock")
            self.source = "mock"

    def get_full_context(self) -> Optional[dict]:
        try:
            if self.source == "bybit":
                return self._get_bybit_context()
            return self._get_mock_context()
        except Exception as e:
            logger.error(f"get_full_context error: {e}", exc_info=True)
            return None

    def _get_bybit_context(self) -> dict:
        """Ambil klines H1 dan M5 dari Bybit."""
        h1_candles = self._fetch_bybit_klines(interval="60", limit=50)
        m5_candles  = self._fetch_bybit_klines(interval="5",  limit=30)

        if not h1_candles or not m5_candles:
            logger.warning("Bybit klines kosong, fallback ke mock")
            return self._get_mock_context()

        # Ticker untuk spread/last price
        ticker = self._fetch_bybit_ticker()
        bid  = float(ticker.get("bid1Price", m5_candles[-1]["close"]))
        ask  = float(ticker.get("ask1Price", m5_candles[-1]["close"] * 1.0001))
        spread_usd = round(ask - bid, 4)

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": self.symbol,
            "current_price": {"bid": bid, "ask": ask, "spread": spread_usd},
            "h1": {"candles": h1_candles},
            "m5":  {"candles": m5_candles},
            "open_positions": [],
        }

    def _fetch_bybit_klines(self, interval: str, limit: int) -> list:
        """Fetch klines dari Bybit v5 API."""
        try:
            resp = self.bybit.get_kline(
                category="linear",
                symbol=self.symbol,
                interval=interval,
                limit=limit,
            )
            if resp["retCode"] != 0:
                logger.error(f"Bybit klines error: {resp['retMsg']}")
                return []

            candles = []
            # Bybit returns: [startTime, open, high, low, close, volume, turnover]
            for row in reversed(resp["result"]["list"]):
                candles.append({
                    "timestamp": datetime.fromtimestamp(
                        int(row[0]) / 1000, tz=timezone.utc
                    ).isoformat(),
                    "open":   float(row[1]),
                    "high":   float(row[2]),
                    "low":    float(row[3]),
                    "close":  float(row[4]),
                    "volume": float(row[5]),
                })
            return candles
        except Exception as e:
            logger.error(f"_fetch_bybit_klines error: {e}")
            return []

    def _fetch_bybit_ticker(self) -> dict:
        try:
            resp = self.bybit.get_tickers(category="linear", symbol=self.symbol)
            if resp["retCode"] == 0 and resp["result"]["list"]:
                return resp["result"]["list"][0]
        except Exception as e:
            logger.error(f"_fetch_ticker error: {e}")
        return {}

    def _get_mock_context(self) -> dict:
        h1_candles = self._generate_mock_candles(50, 60)
        m5_candles  = self._generate_mock_candles(30, 5)
        last = m5_candles[-1]["close"]
        spread = last * 0.00005  # 0.005% spread simulasi

        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": self.symbol,
            "current_price": {
                "bid":    round(last - spread / 2, 4),
                "ask":    round(last + spread / 2, 4),
                "spread": round(spread, 4),
            },
            "h1": {"candles": h1_candles},
            "m5":  {"candles": m5_candles},
            "open_positions": [],
        }

    def _generate_mock_candles(self, count: int, interval_minutes: int) -> list:
        """Buat candle realistis dengan struktur ICT (OB, FVG, BOS)."""
        candles = []
        price = self.mock_price
        trend = random.choice(["bullish", "bearish", "ranging"])
        now = datetime.now(timezone.utc)

        for i in range(count):
            ts = now - timedelta(minutes=interval_minutes * (count - i))
            vol_pct = random.uniform(0.001, 0.005)  # 0.1%–0.5% volatilitas

            if trend == "bullish":
                drift = random.uniform(-0.2, 0.8) * price * vol_pct
            elif trend == "bearish":
                drift = random.uniform(-0.8, 0.2) * price * vol_pct
            else:
                drift = random.uniform(-0.5, 0.5) * price * vol_pct

            # Occasional impulse candle (untuk OB/BOS/FVG)
            if random.random() < 0.08:
                drift *= random.uniform(2.5, 4.0)
                vol_pct *= random.uniform(2.0, 3.5)

            o = price
            c = price + drift
            h = max(o, c) + abs(random.gauss(0, price * vol_pct * 0.3))
            l = min(o, c) - abs(random.gauss(0, price * vol_pct * 0.3))

            candles.append({
                "timestamp": ts.isoformat(),
                "open":   round(o, 4),
                "high":   round(h, 4),
                "low":    round(l, 4),
                "close":  round(c, 4),
                "volume": round(random.uniform(0.5, 50.0), 3),
            })
            price = c

        self.mock_price = price
        return candles
