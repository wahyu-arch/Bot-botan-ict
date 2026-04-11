"""
Risk Manager - 1% risiko per trade (fixed), kalkulasi qty untuk Bybit
"""

import os
import logging

logger = logging.getLogger(__name__)

RISK_PERCENT = 1.0  # FIXED: selalu 1% per trade


class RiskManager:
    def __init__(self):
        self.risk_percent = RISK_PERCENT  # Tidak bisa diubah via env
        self.account_balance = float(os.getenv("ACCOUNT_BALANCE", "10000"))
        self.max_qty = float(os.getenv("MAX_QTY", "1.0"))
        self.min_qty = float(os.getenv("MIN_QTY", "0.001"))

    def calculate_qty(self, entry: float, stop_loss: float, symbol: str) -> float:
        """
        Kalkulasi quantity Bybit berdasarkan 1% risiko akun.

        Formula:
          risk_amount = balance * 1%
          sl_distance = |entry - stop_loss| (dalam USDT)
          qty = risk_amount / sl_distance

        Bybit linear perp (USDT-margined):
          - BTCUSDT: qty dalam BTC, min 0.001
          - ETHUSDT: qty dalam ETH, min 0.01
          - Lainnya: qty dalam koin base
        """
        sl_distance = abs(entry - stop_loss)
        if sl_distance == 0:
            logger.warning("SL distance = 0, menggunakan min qty")
            return self.min_qty

        risk_amount = self.account_balance * (self.risk_percent / 100)
        qty = risk_amount / sl_distance
        qty = round(max(self.min_qty, min(self.max_qty, qty)), 3)

        logger.info(
            f"Risk Calc | Balance: ${self.account_balance:.2f} | "
            f"Risk: {self.risk_percent}% = ${risk_amount:.2f} | "
            f"SL dist: ${sl_distance:.4f} | Qty: {qty}"
        )
        return qty

    def calculate_lot_size(self, entry: float, stop_loss: float, symbol: str) -> float:
        """Alias untuk kompatibilitas dengan trading_bot.py."""
        return self.calculate_qty(entry, stop_loss, symbol)

    def get_max_loss(self) -> float:
        """Maksimum loss per trade dalam USD (selalu 1% balance)."""
        return self.account_balance * (self.risk_percent / 100)
