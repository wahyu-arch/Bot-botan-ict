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

DEFAULT_SYMBOLS = [
    "BTCUSDT",
    "DOGEUSDT",
    "FARTCOINUSDT",
    "XVGUSDT",
    "1000BONKUSDT",
]

# Jeda kecil antar symbol saat startup, supaya tidak hit Groq bersamaan di detik yang sama.
# M5 alignment di bot_core sudah handle timing utama — stagger ini hanya untuk startup.
STAGGER_SECONDS = 10


def _get_symbols() -> list:
    raw = os.getenv("TRADING_SYMBOLS", "")
    if raw:
        return [s.strip().upper() for s in raw.split(",") if s.strip()]
    single = os.getenv("TRADING_SYMBOL", "")
    if single:
        return [single.upper()]
    return DEFAULT_SYMBOLS


async def _run_symbol(symbol: str, stagger: float):
    """Jalankan satu BotCore. Symbol di-pass langsung, stagger via delay."""
    bot = BotCore(symbol=symbol, stagger_delay=stagger)
    await bot.run()


async def run_all(symbols: list):
    tasks = [
        asyncio.create_task(_run_symbol(sym, i * STAGGER_SECONDS))
        for i, sym in enumerate(symbols)
    ]
    await asyncio.gather(*tasks, return_exceptions=True)


def main():
    symbols = _get_symbols()
    threading.Thread(target=run_server, daemon=True).start()

    if len(symbols) == 1:
        bot = BotCore(symbol=symbols[0])
        asyncio.run(bot.run())
    else:
        logger.info(f"Multi-symbol mode: {', '.join(symbols)} | stagger {STAGGER_SECONDS}s")
        asyncio.run(run_all(symbols))


if __name__ == "__main__":
    main()
