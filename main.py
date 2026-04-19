"""Entry point Railway — jalankan Flask API + multi-symbol BotCore paralel."""
import asyncio
import sys
import os
import threading
import logging

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bot_core import BotCore
from api_server import run_server

logger = logging.getLogger(__name__)

# Symbol default. Override via env: TRADING_SYMBOLS=BTCUSDT,DOGEUSDT,FARTCOINUSDT
DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "DOGEUSDT",
    "FARTCOINUSDT",
    "XVGUSDT",
    "1000BONKUSDT",
]


def _get_symbols() -> list:
    raw = os.getenv("TRADING_SYMBOLS", "")
    if raw:
        return [s.strip().upper() for s in raw.split(",") if s.strip()]
    single = os.getenv("TRADING_SYMBOL", "")
    if single:
        return [single.upper()]
    return DEFAULT_SYMBOLS


async def _run_symbol(symbol: str):
    """Jalankan satu BotCore untuk satu symbol — symbol di-pass langsung, bukan lewat env."""
    bot = BotCore(symbol=symbol)   # <-- fix: tidak pakai os.environ race condition
    await bot.run()


async def run_all(symbols: list):
    tasks = [asyncio.create_task(_run_symbol(sym)) for sym in symbols]
    await asyncio.gather(*tasks, return_exceptions=True)


def main():
    symbols = _get_symbols()

    threading.Thread(target=run_server, daemon=True).start()

    if len(symbols) == 1:
        bot = BotCore(symbol=symbols[0])
        asyncio.run(bot.run())
    else:
        logger.info(f"Multi-symbol mode: {', '.join(symbols)}")
        asyncio.run(run_all(symbols))


if __name__ == "__main__":
    main()
