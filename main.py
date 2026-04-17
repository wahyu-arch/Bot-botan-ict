"""Entry point Railway — jalankan Flask API + BotCore."""
import asyncio, sys, os, threading

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from bot_core import BotCore
from api_server import run_server

def main():
    # Flask API di background
    threading.Thread(target=run_server, daemon=True).start()
    # BotCore di main thread
    bot = BotCore()
    asyncio.run(bot.run())

if __name__ == "__main__":
    main()
