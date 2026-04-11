"""
Entry point utama untuk Railway deployment.
"""

import asyncio
import sys
import os

# Fix path agar src/ ditemukan di Railway (/app/main.py)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from trading_bot import ICTTradingBot  # FIX: hapus prefix 'src.'


def main():
    bot = ICTTradingBot()
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
