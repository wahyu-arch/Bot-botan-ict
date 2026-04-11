"""
Entry point utama untuk Railway deployment.
"""

import asyncio
import sys
import os

# Tambah path
sys.path.insert(0, os.path.dirname(__file__))

from src.trading_bot import ICTTradingBot


def main():
    bot = ICTTradingBot()
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
