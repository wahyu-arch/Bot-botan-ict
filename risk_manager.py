"""
Risk Manager - 1% risiko per trade (fixed), kalkulasi qty untuk Bybit Futures
Support leverage: qty dihitung berdasarkan risiko nyata (bukan notional)
"""

import os
import logging

logger = logging.getLogger(__name__)

RISK_PERCENT = 1.0  # FIXED: selalu 1% per trade


class RiskManager:
    def __init__(self):
        self.risk_percent   = RISK_PERCENT
        self.account_balance = float(os.getenv("ACCOUNT_BALANCE", "10000"))
        self.leverage       = int(os.getenv("LEVERAGE", "10"))
        self.max_qty        = float(os.getenv("MAX_QTY", "100.0"))
        self.min_qty        = float(os.getenv("MIN_QTY", "0.001"))

    def calculate_qty(self, entry: float, stop_loss: float, symbol: str) -> float:
        """
        Kalkulasi quantity Bybit Futures berdasarkan 1% risiko akun + leverage.

        Formula dengan leverage:
          risk_amount  = balance * 1%          → uang yang boleh hilang
          sl_distance  = |entry - stop_loss|   → jarak SL dalam USDT per koin
          qty          = risk_amount / sl_distance

        Leverage TIDAK mengubah formula risiko — leverage hanya menentukan
        berapa margin yang dikunci di Bybit. Risiko tetap 1% dari balance.

        Contoh XVG: entry=0.002, SL=0.0018, balance=$18
          risk_amount = $18 * 1% = $0.18
          sl_distance = 0.0002
          qty = 0.18 / 0.0002 = 900 XVGUSDT
          (notional = 900 * 0.002 = $1.8, margin = $1.8 / 10 = $0.18 ✓)
        """
        sl_distance = abs(entry - stop_loss)
        if sl_distance == 0:
            logger.warning("SL distance = 0, menggunakan min qty")
            return self.min_qty

        risk_amount = self.account_balance * (self.risk_percent / 100)
        qty = risk_amount / sl_distance

        # Round ke presisi yang sesuai berdasarkan harga entry
        if entry < 0.01:
            qty = round(qty, 0)   # XVG, SHIB, dll → integer
        elif entry < 1:
            qty = round(qty, 1)
        elif entry < 100:
            qty = round(qty, 2)
        else:
            qty = round(qty, 3)

        qty = max(self.min_qty, min(self.max_qty, qty))

        logger.info(
            f"Risk Calc | {symbol} | Balance: ${self.account_balance:.2f} | "
            f"Leverage: x{self.leverage} | Risk: {self.risk_percent}% = ${risk_amount:.4f} | "
            f"SL dist: {sl_distance:.6f} | Qty: {qty}"
        )
        return qty

    def calculate_lot_size(self, entry: float, stop_loss: float, symbol: str) -> float:
        """Alias untuk kompatibilitas."""
        return self.calculate_qty(entry, stop_loss, symbol)

    def get_max_loss(self) -> float:
        return self.account_balance * (self.risk_percent / 100)
