"""
Memory System - Menyimpan dan mengambil:
- Riwayat trade (win/loss + konteks)
- Error dan pelajaran dari loss
- Iterasi error sementara
- Pattern winning
"""

import json
import os
import logging
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

MEMORY_FILE = "data/trade_memory.json"
ITERATION_ERROR_FILE = "data/iteration_error.json"


class MemorySystem:
    """
    Persistent memory untuk bot trading.
    Menyimpan ke JSON file, cocok untuk Railway.
    """

    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self._ensure_files()

    def _ensure_files(self):
        """Buat file memori jika belum ada."""
        if not os.path.exists(MEMORY_FILE):
            self._save_memory({
                "trades": [],
                "errors": [],
                "stats": {
                    "total_trades": 0,
                    "wins": 0,
                    "losses": 0,
                    "total_pnl": 0.0,
                },
            })

        if not os.path.exists(ITERATION_ERROR_FILE):
            self._save_iteration_error(None)

    def _load_memory(self) -> dict:
        try:
            with open(MEMORY_FILE, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Load memory error: {e}")
            return {"trades": [], "errors": [], "stats": {}}

    def _save_memory(self, data: dict):
        try:
            with open(MEMORY_FILE, "w") as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Save memory error: {e}")

    def _load_iteration_error(self) -> Optional[str]:
        try:
            with open(ITERATION_ERROR_FILE, "r") as f:
                data = json.load(f)
                return data.get("error")
        except Exception:
            return None

    def _save_iteration_error(self, error: Optional[str]):
        try:
            with open(ITERATION_ERROR_FILE, "w") as f:
                json.dump({"error": error, "timestamp": datetime.now(timezone.utc).isoformat()}, f)
        except Exception as e:
            logger.error(f"Save iteration error: {e}")

    def log_trade(
        self,
        symbol: str,
        direction: str,
        setup: str,
        entry: float,
        sl: float,
        tp: float,
        rr: float,
        confidence: float,
        notes: str,
        trade_id: Optional[str] = None,
    ):
        """Catat trade baru ke memori."""
        memory = self._load_memory()

        trade = {
            "trade_id": trade_id or f"T{len(memory['trades']) + 1:04d}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "symbol": symbol,
            "direction": direction,
            "setup": setup,
            "entry": entry,
            "sl": sl,
            "tp": tp,
            "rr": rr,
            "confidence": confidence,
            "notes": notes,
            "result": "open",
            "pnl": 0.0,
            "exit_price": None,
            "exit_reason": None,
        }

        memory["trades"].append(trade)
        memory["stats"]["total_trades"] += 1
        self._save_memory(memory)
        logger.info(f"Trade logged: {trade['trade_id']}")

    def update_trade_result(
        self,
        trade_id: str,
        result: str,
        pnl: float,
        exit_price: float,
        exit_reason: str,
    ):
        """Update hasil trade yang sudah closed."""
        memory = self._load_memory()

        for trade in memory["trades"]:
            if trade["trade_id"] == trade_id:
                trade["result"] = result
                trade["pnl"] = pnl
                trade["exit_price"] = exit_price
                trade["exit_reason"] = exit_reason
                trade["closed_at"] = datetime.now(timezone.utc).isoformat()

                # Update stats
                if result == "win":
                    memory["stats"]["wins"] += 1
                elif result == "loss":
                    memory["stats"]["losses"] += 1
                memory["stats"]["total_pnl"] += pnl
                break

        self._save_memory(memory)

    def log_error(self, error: str, lesson: str, context: dict = None):
        """Catat error dan pelajaran untuk self-iteration."""
        memory = self._load_memory()

        error_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "error": error,
            "lesson": lesson,
            "context": context or {},
        }

        memory["errors"].append(error_entry)

        # Simpan hanya 50 error terakhir
        if len(memory["errors"]) > 50:
            memory["errors"] = memory["errors"][-50:]

        self._save_memory(memory)
        logger.info(f"Error logged: {error[:80]}...")

    def log_iteration_error(self, error: str):
        """Simpan error iterasi sementara untuk self-correction."""
        self._save_iteration_error(error)

    def get_last_iteration_error(self) -> Optional[str]:
        """Ambil error dari iterasi terakhir."""
        return self._load_iteration_error()

    def clear_iteration_error(self):
        """Bersihkan error iterasi setelah siklus selesai."""
        self._save_iteration_error(None)

    def get_recent_errors(self, limit: int = 5) -> list:
        """Ambil error terbaru untuk konteks AI."""
        memory = self._load_memory()
        return memory.get("errors", [])[-limit:]

    def get_recent_trades(self, result: str = None, limit: int = 10) -> list:
        """Ambil trade terbaru, opsional filter by result."""
        memory = self._load_memory()
        trades = memory.get("trades", [])

        if result:
            trades = [t for t in trades if t.get("result") == result]

        return trades[-limit:]

    def get_stats(self) -> dict:
        """Ambil statistik performa keseluruhan."""
        memory = self._load_memory()
        stats = memory.get("stats", {})
        total = stats.get("total_trades", 0)
        wins = stats.get("wins", 0)

        return {
            "total_trades": total,
            "wins": wins,
            "losses": stats.get("losses", 0),
            "open_trades": len([t for t in memory.get("trades", []) if t.get("result") == "open"]),
            "win_rate": wins / total if total > 0 else 0,
            "total_pnl": stats.get("total_pnl", 0.0),
            "total_errors": len(memory.get("errors", [])),
        }

    def get_losing_patterns(self) -> list:
        """Identifikasi pola yang sering loss untuk dihindari."""
        memory = self._load_memory()
        losses = [t for t in memory.get("trades", []) if t.get("result") == "loss"]

        # Kelompokkan setup yang sering loss
        setup_counts = {}
        for trade in losses:
            setup = trade.get("setup", "unknown")
            setup_counts[setup] = setup_counts.get(setup, 0) + 1

        return sorted(
            [{"setup": k, "loss_count": v} for k, v in setup_counts.items()],
            key=lambda x: x["loss_count"],
            reverse=True,
        )[:5]

    def export_summary(self) -> str:
        """Export ringkasan memori untuk debugging."""
        stats = self.get_stats()
        recent_errors = self.get_recent_errors(3)
        losing_patterns = self.get_losing_patterns()

        summary = f"""
=== MEMORY SUMMARY ===
Total Trades: {stats['total_trades']}
Win Rate: {stats['win_rate']:.0%} ({stats['wins']}W / {stats['losses']}L)
Total PnL: {stats['total_pnl']:.2f}
Open Trades: {stats['open_trades']}
Errors Logged: {stats['total_errors']}

Recent Errors:
{json.dumps(recent_errors, indent=2, default=str)}

Losing Patterns:
{json.dumps(losing_patterns, indent=2)}
"""
        return summary
