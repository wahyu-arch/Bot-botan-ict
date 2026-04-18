"""
WatchlistEngine — Sistem trigger berbasis harga.

Konsep:
- Bot TIDAK analisis AI setiap siklus.
- AI hanya dipanggil saat ada "trigger" terjadi:
    1. Sesi pertama kali: AI diskusi kondisi awal, set watchlist
    2. Saat harga sentuh/break level yang AI tentukan: AI diskusi lagi
    3. Begitu seterusnya sampai konfirmasi entry atau invalidasi

Siklus bot (setiap SCAN_INTERVAL detik):
  → Ambil harga terbaru
  → Cek apakah ada watchlist yang tersentuh
  → Jika ya: panggil AI diskusi
  → Jika tidak: diam, log saja harga

Struktur watchlist item:
{
  "id": "wl_20260412_130001",
  "level": 73450.5,           # harga trigger (presisi dari data)
  "condition": "touch|break_above|break_below",
  "reason": "BOS confirmation level",
  "phase": "waiting_bos|waiting_entry|waiting_retest",
  "triggered": False,
  "created_at": "...",
  "triggered_at": None,
  "session_ref": "20260412_130001",  # sesi diskusi yang set watchlist ini
}
"""

import os
import json
import logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

WATCHLIST_FILE = "data/watchlist.json"


class WatchlistEngine:
    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self.items: list = self._load()

    def _load(self) -> list:
        if os.path.exists(WATCHLIST_FILE):
            try:
                with open(WATCHLIST_FILE) as f:
                    return json.load(f)
            except Exception:
                pass
        return []

    def _save(self):
        with open(WATCHLIST_FILE, "w") as f:
            json.dump(self.items, f, indent=2, ensure_ascii=False)

    def add(self, level: float, condition: str, reason: str, phase: str, session_ref: str, symbol: str = "") -> dict:
        """Tambah item watchlist baru."""
        item = {
            "id": f"wl_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}",
            "level": round(level, 2),
            "condition": condition,
            "reason": reason,
            "phase": phase,
            "triggered": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "triggered_at": None,
            "session_ref": session_ref,
            "symbol": symbol,
        }
        self.items.append(item)
        self._save()
        logger.info(
            f"[WATCHLIST] +Tambah | {condition.upper()} @ {level:.2f} | "
            f"Phase: {phase} | Reason: {reason}"
        )
        return item

    def add_many(self, levels: list, session_ref: str):
        """Tambah banyak watchlist sekaligus dari hasil diskusi AI."""
        for lvl in levels:
            self.add(
                level=lvl.get("level", 0),
                condition=lvl.get("condition", "touch"),
                reason=lvl.get("reason", ""),
                phase=lvl.get("phase", "waiting_bos"),
                session_ref=session_ref,
            )

    def clear_untriggered(self):
        """Hapus semua watchlist yang belum trigger (reset setelah sesi baru)."""
        before = len(self.items)
        self.items = [i for i in self.items if i.get("triggered")]
        self._save()
        logger.info(f"[WATCHLIST] Cleared {before - len(self.items)} untriggered items")

    def check(self, current_price: float, prev_price: float) -> list:
        """
        Cek semua watchlist aktif. Return list item yang baru trigger.
        Gunakan high/low candle terakhir untuk presisi: pakai current_price sebagai proxy.
        """
        triggered_now = []
        for item in self.items:
            if item["triggered"]:
                continue
            level = item["level"]
            cond  = item["condition"]

            fired = False
            if cond == "touch":
                # Harga menyentuh atau melewati level (dari arah manapun)
                fired = (
                    (prev_price < level <= current_price) or
                    (prev_price > level >= current_price)
                )
            elif cond == "break_above":
                # Close/harga naik melewati level
                fired = prev_price <= level < current_price
            elif cond == "break_below":
                # Close/harga turun melewati level
                fired = prev_price >= level > current_price

            if fired:
                item["triggered"] = True
                item["triggered_at"] = datetime.now(timezone.utc).isoformat()
                item["triggered_price"] = round(current_price, 2)
                triggered_now.append(item)
                logger.info(
                    f"[WATCHLIST] 🔔 TRIGGER | {cond.upper()} @ {level:.2f} | "
                    f"Harga: {current_price:.2f} | Phase: {item['phase']} | {item['reason']}"
                )

        if triggered_now:
            self._save()

        return triggered_now

    def get_active(self) -> list:
        return [i for i in self.items if not i["triggered"]]

    def summary(self) -> str:
        active = self.get_active()
        if not active:
            return "Watchlist kosong"
        lines = [f"  {i['condition'].upper()} @ {i['level']:.2f} [{i['phase']}] — {i['reason']}"
                 for i in active]
        return "\n".join(lines)

    def to_api_dict(self) -> list:
        # Kirim semua aktif + 5 terakhir yang triggered (untuk history)
        active = [i for i in self.items if not i.get('triggered')]
        triggered = [i for i in self.items if i.get('triggered')][-5:]
        return active + triggered
