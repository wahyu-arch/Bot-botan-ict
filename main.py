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

# Daftar symbol default yang dijalankan paralel.
# Override via env: TRADING_SYMBOLS=BTCUSDT,DOGEUSDT,FARTCOINUSDT
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
    # Fallback ke TRADING_SYMBOL tunggal (kompatibilitas v28 ke bawah)
    single = os.getenv("TRADING_SYMBOL", "")
    if single:
        return [single.upper()]
    return DEFAULT_SYMBOLS


async def _run_symbol(symbol: str):
    """Jalankan satu BotCore untuk satu symbol."""
    os.environ["TRADING_SYMBOL"] = symbol
    bot = BotCore()
    await bot.run()


async def run_all(symbols: list):
    """Jalankan semua symbol secara concurrent dalam satu event loop."""
    tasks = [asyncio.create_task(_run_symbol(sym)) for sym in symbols]
    await asyncio.gather(*tasks, return_exceptions=True)


def main():
    symbols = _get_symbols()

    # Flask API di background thread
    threading.Thread(target=run_server, daemon=True).start()

    if len(symbols) == 1:
        # Single symbol — jalankan langsung seperti sebelumnya
        os.environ["TRADING_SYMBOL"] = symbols[0]
        bot = BotCore()
        asyncio.run(bot.run())
    else:
        # Multi-symbol — semua jalan paralel dalam satu event loop
        logger.info(f"Multi-symbol mode aktif: {', '.join(symbols)}")
        asyncio.run(run_all(symbols))


if __name__ == "__main__":
    main()
