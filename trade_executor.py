"""
Trade Executor - Bybit API (Linear Perpetual USDT)
Paper trading + Live via pybit v5
"""

import os
import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

PAPER_TRADES_FILE = "data/paper_trades.json"


class TradeExecutor:
    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        os.makedirs("data", exist_ok=True)

        if paper_mode:
            self._ensure_paper_file()
            logger.info("Trade Executor: PAPER MODE aktif")
        else:
            self._init_bybit()
            logger.info("Trade Executor: BYBIT LIVE aktif")

    def _ensure_paper_file(self):
        if not os.path.exists(PAPER_TRADES_FILE):
            with open(PAPER_TRADES_FILE, "w") as f:
                json.dump({"open": [], "closed": []}, f)

    def _init_bybit(self):
        """Inisialisasi Bybit via pybit v5."""
        bybit_key = os.environ.get("BYBIT_API_KEY")
        bybit_secret = os.environ.get("BYBIT_API_SECRET")
        if not bybit_key or not bybit_secret:
            raise EnvironmentError(
                "[FATAL] BYBIT_API_KEY atau BYBIT_API_SECRET tidak ditemukan! "
                "Tambahkan di Railway: Settings > Variables"
            )
        try:
            from pybit.unified_trading import HTTP
            self.bybit = HTTP(
                testnet=os.getenv("BYBIT_TESTNET", "true").lower() == "true",
                api_key=bybit_key,
                api_secret=bybit_secret,
            )
            # Verifikasi koneksi
            resp = self.bybit.get_wallet_balance(accountType="UNIFIED")
            if resp["retCode"] != 0:
                raise RuntimeError(f"Bybit auth failed: {resp['retMsg']}")
            logger.info("Bybit connection verified ✓")
        except ImportError:
            logger.warning("pybit tidak terinstall, fallback ke paper mode")
            self.paper_mode = True
            self._ensure_paper_file()

    def get_account_balance(self) -> float:
        """Ambil balance USDT dari Bybit."""
        if self.paper_mode:
            return float(os.getenv("ACCOUNT_BALANCE", "10000"))
        try:
            resp = self.bybit.get_wallet_balance(accountType="UNIFIED", coin="USDT")
            if resp["retCode"] == 0:
                coins = resp["result"]["list"][0]["coin"]
                for coin in coins:
                    if coin["coin"] == "USDT":
                        bal = float(coin["walletBalance"])
                        logger.info(f"Bybit balance: ${bal:.2f} USDT")
                        return bal
        except Exception as e:
            logger.error(f"get_balance error: {e}")
        return float(os.getenv("ACCOUNT_BALANCE", "10000"))

    def execute(
        self,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        lot_size: float,  # qty dalam koin Bybit
        signal: dict,
    ) -> dict:
        """Eksekusi order ke Bybit atau paper."""
        trade_id = str(uuid.uuid4())[:8].upper()

        if self.paper_mode:
            return self._paper_execute(
                trade_id, direction, entry_price, stop_loss, take_profit, lot_size, signal
            )
        return self._bybit_execute(
            trade_id, direction, entry_price, stop_loss, take_profit, lot_size
        )

    def _bybit_execute(
        self,
        trade_id: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        qty: float,
    ) -> dict:
        """
        Kirim market order ke Bybit Linear Perpetual.
        Menggunakan orderLinkId agar idempoten.
        """
        symbol = os.getenv("TRADING_SYMBOL", "BTCUSDT")
        side = "Buy" if direction == "buy" else "Sell"

        try:
            resp = self.bybit.place_order(
                category="linear",
                symbol=symbol,
                side=side,
                orderType="Market",
                qty=str(qty),
                stopLoss=str(round(stop_loss, 4)),
                takeProfit=str(round(take_profit, 4)),
                slTriggerBy="MarkPrice",
                tpTriggerBy="MarkPrice",
                timeInForce="IOC",
                orderLinkId=trade_id,
                reduceOnly=False,
            )

            if resp["retCode"] == 0:
                bybit_order_id = resp["result"]["orderId"]
                logger.info(
                    f"BYBIT ORDER #{bybit_order_id}: {side} {qty} {symbol} "
                    f"@ mkt | SL: {stop_loss} | TP: {take_profit}"
                )
                return {"trade_id": bybit_order_id, "status": "filled", "link_id": trade_id}
            else:
                err = resp["retMsg"]
                logger.error(f"Bybit order rejected: {err}")
                return {"trade_id": trade_id, "status": "failed", "error": err}

        except Exception as e:
            logger.error(f"Bybit execution error: {e}")
            return {"trade_id": trade_id, "status": "failed", "error": str(e)}

    def _paper_execute(
        self,
        trade_id: str,
        direction: str,
        entry_price: float,
        stop_loss: float,
        take_profit: float,
        qty: float,
        signal: dict,
    ) -> dict:
        try:
            with open(PAPER_TRADES_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            data = {"open": [], "closed": []}

        trade = {
            "trade_id": trade_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": os.getenv("TRADING_SYMBOL", "BTCUSDT"),
            "direction": direction,
            "entry_price": entry_price,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "qty": qty,
            "sl_distance_usd": round(abs(entry_price - stop_loss) * qty, 4),
            "max_loss_usd": round(abs(entry_price - stop_loss) * qty, 2),
            "bias": signal.get("bias_m15", ""),
            "setup": signal.get("entry_reason", ""),
            "confidence": signal.get("confidence", 0),
            "status": "open",
        }

        data["open"].append(trade)
        with open(PAPER_TRADES_FILE, "w") as f:
            json.dump(data, f, indent=2)

        logger.info(
            f"PAPER #{trade_id}: {direction.upper()} {qty} @ {entry_price} | "
            f"SL: {stop_loss} | TP: {take_profit} | Max loss: ${trade['max_loss_usd']:.2f}"
        )
        return {"trade_id": trade_id, "status": "filled"}

    def check_closed_trades(self) -> list:
        """Cek trade yang sudah hit SL/TP."""
        if self.paper_mode:
            return self._check_paper_closed()

        # Bybit: query closed P&L
        try:
            symbol = os.getenv("TRADING_SYMBOL", "BTCUSDT")
            resp = self.bybit.get_closed_pnl(category="linear", symbol=symbol, limit=10)
            if resp["retCode"] == 0:
                closed = []
                for item in resp["result"]["list"]:
                    pnl = float(item.get("closedPnl", 0))
                    closed.append({
                        "trade_id": item["orderId"],
                        "result": "win" if pnl > 0 else "loss",
                        "pnl": pnl,
                        "exit_price": float(item.get("avgExitPrice", 0)),
                        "exit_reason": "tp_sl",
                    })
                return closed
        except Exception as e:
            logger.error(f"check_closed error: {e}")
        return []

    def _check_paper_closed(self) -> list:
        """Simulasi SL/TP hit untuk paper trading."""
        import random
        try:
            with open(PAPER_TRADES_FILE, "r") as f:
                data = json.load(f)
        except Exception:
            return []

        closed_now = []
        still_open = []

        for trade in data.get("open", []):
            # 5% chance per scan untuk simulasi close
            if random.random() < 0.05:
                result = random.choices(["win", "loss"], weights=[0.55, 0.45])[0]
                if result == "win":
                    exit_price = trade["take_profit"]
                    pnl = abs(trade["take_profit"] - trade["entry_price"]) * trade["qty"]
                else:
                    exit_price = trade["stop_loss"]
                    pnl = -abs(trade["entry_price"] - trade["stop_loss"]) * trade["qty"]

                closed_trade = {
                    **trade,
                    "result": result,
                    "pnl": round(pnl, 2),
                    "exit_price": exit_price,
                    "exit_reason": "tp_sl_hit",
                    "closed_at": datetime.now(timezone.utc).isoformat(),
                }
                closed_now.append(closed_trade)
                data.setdefault("closed", []).append(closed_trade)
                logger.info(f"Paper closed #{trade['trade_id']}: {result.upper()} PnL ${pnl:.2f}")
            else:
                still_open.append(trade)

        data["open"] = still_open
        with open(PAPER_TRADES_FILE, "w") as f:
            json.dump(data, f, indent=2)

        return closed_now

    def get_open_positions(self) -> list:
        """Ambil posisi aktif dari Bybit."""
        if self.paper_mode:
            try:
                with open(PAPER_TRADES_FILE, "r") as f:
                    return json.load(f).get("open", [])
            except Exception:
                return []
        try:
            symbol = os.getenv("TRADING_SYMBOL", "BTCUSDT")
            resp = self.bybit.get_positions(category="linear", symbol=symbol)
            if resp["retCode"] == 0:
                return [
                    p for p in resp["result"]["list"]
                    if float(p.get("size", 0)) > 0
                ]
        except Exception as e:
            logger.error(f"get_positions error: {e}")
        return []
