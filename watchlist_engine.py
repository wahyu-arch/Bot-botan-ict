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


def _smart_decimals(price: float) -> int:
    """Tentukan jumlah desimal berdasarkan magnitude harga."""
    if price <= 0:
        return 6
    if price >= 1000:   return 2   # BTC, ETH
    if price >= 10:     return 3
    if price >= 1:      return 4
    if price >= 0.1:    return 5
    if price >= 0.01:   return 6
    return 8                        # sangat kecil (SHIB, dll)


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

    def add(self, level: float, condition: str, reason: str, phase: str, session_ref: str,
             symbol: str = "", assigned_to: str = "", action: str = "",
             expires_on_phase_change: bool = True, ttl_hours: float = 24.0) -> dict:
        """Tambah item watchlist baru.
        assigned_to: AI yang dipanggil saat trigger (hiura/senanan/shina/yusuf/katyusha/auto)
        action: aksi spesifik saat trigger (re_analyze/check_bos/check_mss/entry/alert)
        expires_on_phase_change: hapus otomatis kalau phase berubah
        ttl_hours: hapus otomatis setelah N jam (0 = tidak expire)
        """
        item = {
            "id": f"wl_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S_%f')[:22]}",
            "level": round(level, _smart_decimals(level)),
            "condition": condition,
            "reason": reason,
            "phase": phase,
            "triggered": False,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "triggered_at": None,
            "session_ref": session_ref,
            "symbol": symbol,
            "assigned_to": assigned_to,
            "action": action,
            "expires_on_phase_change": expires_on_phase_change,
            "ttl_hours": ttl_hours,
        }
        self.items.append(item)
        self._save()
        logger.info(
            f"[WATCHLIST] +Tambah | {condition.upper()} @ {level} | "
            f"Phase: {phase} | Reason: {reason} | TTL: {ttl_hours}h"
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

    def expire_stale(self, current_phase: str):
        """
        Hapus watchlist yang:
        1. expires_on_phase_change=True dan phase berubah
        2. Sudah melewati ttl_hours sejak dibuat
        """
        now = datetime.now(timezone.utc)
        before = len(self.items)
        kept = []
        for item in self.items:
            if item.get("triggered"):
                kept.append(item)
                continue
            # Cek TTL
            ttl = item.get("ttl_hours", 24.0)
            if ttl > 0:
                try:
                    created = datetime.fromisoformat(item["created_at"])
                    age_hours = (now - created).total_seconds() / 3600
                    if age_hours > ttl:
                        logger.info(f"[WATCHLIST] Expired (TTL {ttl}h) | {item['condition']} @ {item['level']}")
                        continue
                except Exception:
                    pass
            # Cek phase change
            if item.get("expires_on_phase_change") and item.get("phase") != current_phase:
                # Hanya expire kalau phase item spesifik dan sudah tidak relevan
                item_phase = item.get("phase", "")
                # Fase yang sudah lewat: kalau sekarang fvg_wait, hapus h1_scan watchlist
                phase_order = ["h1_scan","fvg_wait","bos_guard","entry_sniper"]
                try:
                    current_idx = phase_order.index(current_phase)
                    item_idx    = phase_order.index(item_phase)
                    if item_idx < current_idx - 1:  # lebih dari 1 fase ke belakang
                        logger.info(f"[WATCHLIST] Expired (phase stale: {item_phase}→{current_phase}) | @ {item['level']}")
                        continue
                except ValueError:
                    pass
            kept.append(item)

        removed = before - len(kept)
        if removed:
            self.items = kept
            self._save()
            logger.info(f"[WATCHLIST] expire_stale: hapus {removed} item kadaluarsa")

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
                    f"[WATCHLIST] 🔔 TRIGGER | {cond.upper()} @ {level} | "
                    f"Harga: {current_price} | Phase: {item['phase']} | {item['reason']}"
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
        lines = [f"  {i['condition'].upper()} @ {i['level']} [{i['phase']}] — {i['reason']}"
                 for i in active]
        return "\n".join(lines)

    def to_api_dict(self) -> list:
        # Kirim semua aktif + 5 terakhir yang triggered (untuk history)
        active = [i for i in self.items if not i.get('triggered')]
        triggered = [i for i in self.items if i.get('triggered')][-5:]
        return active + triggered
