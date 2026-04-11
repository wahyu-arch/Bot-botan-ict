"""
Entry point utama untuk Railway deployment.
Menjalankan Flask API server + Trading Bot secara paralel.
"""

import asyncio
import sys
import os
import threading

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from trading_bot import ICTTradingBot
from api_server import run_server


def main():
    # Jalankan Flask API di background thread
    api_thread = threading.Thread(target=run_server, daemon=True)
    api_thread.start()

    # Jalankan bot di main thread
    bot = ICTTradingBot()
    asyncio.run(bot.run())


if __name__ == "__main__":
    main()
