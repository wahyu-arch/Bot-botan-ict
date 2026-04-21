"""
CandleReplay — Replay candle satu per satu kiri ke kanan.

Bukan analisis semua sekaligus. Bot "hidup" di setiap candle,
berhenti saat kondisi terpenuhi, lanjut dari sana.

State machine:
  SCAN_BOS     → replay candle H1 satu per satu, cari swing + BOS
  WAIT_FVG     → BOS terbentuk, pantau candle baru, tunggu FVG disentuh
  IDM_HUNT     → FVG disentuh, replay M5 cari IDM
  BOS_GUARD    → IDM disentuh, tunggu BOS/MSS M5
  ENTRY        → MSS terkonfirmasi, hitung entry
"""

import logging
logger = logging.getLogger(__name__)


# ── Field helpers (support h/high, l/low, c/close) ─────

def _h(c): return c.get("high", c.get("h", 0))
def _l(c): return c.get("low",  c.get("l", 0))
def _c(c): return c.get("close",c.get("c", 0))
def _o(c): return c.get("open", c.get("o", 0))
def _ts(c): return c.get("ts", "")


# ── Swing detection ─────────────────────────────────────

def is_swing_high(candles: list, idx: int, left: int, right: int) -> bool:
    """Candle idx adalah swing high jika lebih tinggi dari L kiri dan R kanan."""
    h = _h(candles[idx])
    left_ok  = all(_h(candles[j]) < h for j in range(max(0, idx-left), idx))
    right_ok = all(_h(candles[j]) < h for j in range(idx+1, min(len(candles), idx+right+1)))
    return left_ok and right_ok and len(range(max(0,idx-left),idx)) == left and \
           len(range(idx+1, min(len(candles), idx+right+1))) == right


def is_swing_low(candles: list, idx: int, left: int, right: int) -> bool:
    """Candle idx adalah swing low jika lebih rendah dari L kiri dan R kanan."""
    l = _l(candles[idx])
    left_ok  = all(_l(candles[j]) > l for j in range(max(0, idx-left), idx))
    right_ok = all(_l(candles[j]) > l for j in range(idx+1, min(len(candles), idx+right+1)))
    return left_ok and right_ok and len(range(max(0,idx-left),idx)) == left and \
           len(range(idx+1, min(len(candles), idx+right+1))) == right


# ── FVG detection ────────────────────────────────────────

def find_fvg_at(candles: list, idx: int, direction: str, min_gap_pct: float = 0.05) -> dict | None:
    """
    Cek apakah ada FVG yang terbentuk di candle idx (C1=idx-2, C2=idx-1, C3=idx).
    FVG bearish: C1.low > C3.high
    FVG bullish: C1.high < C3.low
    """
    if idx < 2:
        return None
    c1, c3 = candles[idx-2], candles[idx]
    if direction == "bearish":
        if _l(c1) > _h(c3):
            high = _l(c1)
            low  = _h(c3)
            mid  = (high + low) / 2
            gap_pct = (high - low) / mid * 100
            if gap_pct >= min_gap_pct:
                return {"type": "bearish", "high": round(high,8), "low": round(low,8),
                        "mid": round(mid,8), "gap_pct": round(gap_pct,4),
                        "formed_at": idx, "ts": _ts(c1), "filled": False}
    elif direction == "bullish":
        if _h(c1) < _l(c3):
            low  = _h(c1)
            high = _l(c3)
            mid  = (high + low) / 2
            gap_pct = (high - low) / mid * 100
            if gap_pct >= min_gap_pct:
                return {"type": "bullish", "high": round(high,8), "low": round(low,8),
                        "mid": round(mid,8), "gap_pct": round(gap_pct,4),
                        "formed_at": idx, "ts": _ts(c1), "filled": False}
    return None


def check_fvg_filled(fvg: dict, candle: dict) -> bool:
    """
    FVG filled = ada candle yang CLOSE menembus zona FVG.
    Wick boleh masuk (wick touch = belum filled).
    """
    cl = _c(candle)
    if fvg["type"] == "bearish":
        return cl > fvg["high"]   # close di atas FVG bearish = filled
    else:
        return cl < fvg["low"]    # close di bawah FVG bullish = filled


def check_fvg_touched(fvg: dict, candle: dict) -> bool:
    """
    FVG touched = wick ATAU body masuk ke zona FVG.
    Kalau harga sudah di dalam zona (antara low dan high), langsung touched.
    Kalau harga menyentuh batas zona dari luar, juga touched.
    """
    h, l = _h(candle), _l(candle)
    fvg_low, fvg_high = fvg["low"], fvg["high"]

    # Harga di dalam zona (body atau wick overlap dengan FVG)
    # Overlap = h >= fvg_low AND l <= fvg_high
    return h >= fvg_low and l <= fvg_high


# ── IDM detection (state machine) ───────────────────────

def find_idm_replay(candles: list, direction: str,
                    gap_min: int = 1, max_search: int = 50) -> dict | None:
    """
    Replay M5 kiri ke kanan cari IDM.
    Berhenti saat IDM terbentuk.
    """
    n = len(candles)
    for i in range(1, n):
        ca = candles[i]
        if direction == "bearish":
            high_a = _h(ca)
            if high_a <= _h(candles[i-1]):
                continue
            gap = 0
            tembus = None
            for j in range(i+1, min(i+max_search, n)):
                cj_h = _h(candles[j])
                cj_c = _c(candles[j])
                if cj_h >= high_a or cj_c >= high_a:
                    if gap >= gap_min:
                        tembus = j
                    break
                gap += 1
            if tembus:
                return {"type":"bearish_idm","candle_a_idx":i,
                        "candle_a_high":round(high_a,8),
                        "candle_a_low":round(_l(ca),8),
                        "watch_level":round(_l(ca),8),
                        "tembus_idx":tembus,"gap":gap,"ts":_ts(ca)}
        elif direction == "bullish":
            low_a = _l(ca)
            if low_a >= _l(candles[i-1]):
                continue
            gap = 0
            tembus = None
            for j in range(i+1, min(i+max_search, n)):
                cj_l = _l(candles[j])
                cj_c = _c(candles[j])
                if cj_l <= low_a or cj_c <= low_a:
                    if gap >= gap_min:
                        tembus = j
                    break
                gap += 1
            if tembus:
                return {"type":"bullish_idm","candle_a_idx":i,
                        "candle_a_high":round(_h(ca),8),
                        "candle_a_low":round(low_a,8),
                        "watch_level":round(_h(ca),8),
                        "tembus_idx":tembus,"gap":gap,"ts":_ts(ca)}
    return None


# ── Main replay engine ───────────────────────────────────

class ReplayEngine:
    """
    Replay candle H1 satu per satu kiri ke kanan.
    Berhenti saat BOS atau FVG touch ditemukan.
    State disimpan dan dilanjutkan di siklus berikutnya.
    """

    def __init__(self, sw_left=8, sw_right=8, min_gap_pct=0.05):
        self.sw_left     = sw_left
        self.sw_right    = sw_right
        self.min_gap_pct = min_gap_pct

        # State — disimpan antar siklus
        self.state       = "SCAN_BOS"
        self.swing_highs : list[tuple] = []   # (idx, price)
        self.swing_lows  : list[tuple] = []   # (idx, price)
        self.bos         : dict | None = None
        self.fvgs        : list[dict]  = []   # FVG watchlist
        self.last_h1_idx : int = 0            # idx terakhir yang sudah diproses
        self.idm_m5      : dict | None = None
        self.idm_watch   : float = 0.0

    def _update_swings(self, candles: list, up_to: int):
        """Update swing list sampai candle up_to (pastikan swing_right terpenuhi)."""
        for i in range(self.last_h1_idx, up_to - self.sw_right):
            if i < self.sw_left:
                continue
            if is_swing_high(candles, i, self.sw_left, self.sw_right):
                self.swing_highs.append((i, _h(candles[i])))
            if is_swing_low(candles, i, self.sw_left, self.sw_right):
                self.swing_lows.append((i, _l(candles[i])))

    def replay_h1(self, candles: list, current_price: float = 0) -> dict:
        """
        Replay semua candle H1, lanjut dari posisi terakhir.
        Return: {"event": "bos"|"fvg_touched"|"none", "data": {...}}
        """
        n = len(candles)

        # ── STATE: SCAN_BOS ──────────────────────────────
        if self.state == "SCAN_BOS":
            for i in range(self.last_h1_idx, n):
                # Update swing saat candle ke-i sudah punya cukup candle kanan
                self._update_swings(candles, i)

                c = candles[i]
                cl = _c(c)

                # Cek BOS bearish: close di bawah swing low valid terakhir
                if self.swing_lows:
                    last_sl_idx, last_sl_price = self.swing_lows[-1]
                    if last_sl_idx + self.sw_right < i and cl < last_sl_price:
                        # BOS bearish terkonfirmasi!
                        sh_before = max((p for _,p in self.swing_highs if _ < last_sl_idx), default=_h(c))
                        sh_since  = max((_h(candles[j]) for j in range(last_sl_idx, i+1)), default=_h(c))
                        self.bos = {
                            "type": "bearish_bos", "direction": "bearish",
                            "swing_level": round(last_sl_price, 8),
                            "swing_idx": last_sl_idx,
                            "bos_candle_idx": i,
                            "bos_close": round(cl, 8),
                            "bos_ts": _ts(c),
                            "sh_before_bos": round(sh_before, 8),
                            "sh_since_bos": round(sh_since, 8),
                            "sl_since_bos": round(min(_l(candles[j]) for j in range(last_sl_idx,i+1)), 8),
                        }
                        # Cari FVG bearish setelah BOS
                        self.fvgs = []
                        for fi in range(last_sl_idx, i+1):
                            fvg = find_fvg_at(candles, fi, "bearish", self.min_gap_pct)
                            if fvg:
                                self.fvgs.append(fvg)
                        self.fvgs = self.fvgs[:3]  # max 3
                        self.last_h1_idx = i + 1
                        self.state = "WAIT_FVG"
                        logger.info(f"[REPLAY] BOS bearish @ {last_sl_price} | FVG: {len(self.fvgs)}")
                        return {"event": "bos", "data": {"bos": self.bos, "fvgs": self.fvgs}}

                # Cek BOS bullish
                if self.swing_highs:
                    last_sh_idx, last_sh_price = self.swing_highs[-1]
                    if last_sh_idx + self.sw_right < i and cl > last_sh_price:
                        sl_before = min((p for _,p in self.swing_lows if _ < last_sh_idx), default=_l(c))
                        sl_since  = min((_l(candles[j]) for j in range(last_sh_idx, i+1)), default=_l(c))
                        self.bos = {
                            "type": "bullish_bos", "direction": "bullish",
                            "swing_level": round(last_sh_price, 8),
                            "swing_idx": last_sh_idx,
                            "bos_candle_idx": i,
                            "bos_close": round(cl, 8),
                            "bos_ts": _ts(c),
                            "sl_before_bos": round(sl_before, 8),
                            "sl_since_bos": round(sl_since, 8),
                            "sh_since_bos": round(max(_h(candles[j]) for j in range(last_sh_idx,i+1)), 8),
                        }
                        self.fvgs = []
                        for fi in range(last_sh_idx, i+1):
                            fvg = find_fvg_at(candles, fi, "bullish", self.min_gap_pct)
                            if fvg:
                                self.fvgs.append(fvg)
                        self.fvgs = self.fvgs[:3]
                        self.last_h1_idx = i + 1
                        self.state = "WAIT_FVG"
                        logger.info(f"[REPLAY] BOS bullish @ {last_sh_price} | FVG: {len(self.fvgs)}")
                        return {"event": "bos", "data": {"bos": self.bos, "fvgs": self.fvgs}}

            self.last_h1_idx = max(0, n - self.sw_right - 1)
            return {"event": "none", "data": {}}

        # ── STATE: WAIT_FVG ──────────────────────────────
        elif self.state == "WAIT_FVG":
            bos_dir = self.bos.get("direction","")
            sw_ref  = self.bos.get("swing_level", 0)

            # Cek langsung: kalau harga saat ini sudah di dalam FVG, trigger immediately
            # Gunakan current_price (harga live bid) bukan candle terakhir yang sudah close
            if current_price > 0:
                price_candle = {"h": current_price, "l": current_price, "c": current_price, "o": current_price}
            else:
                price_candle = candles[-1] if candles else None

            if price_candle:
                for fvg in self.fvgs:
                    if check_fvg_touched(fvg, price_candle) and not check_fvg_filled(fvg, price_candle):
                        self.state = "IDM_HUNT"
                        self.last_h1_idx = n
                        p = current_price or _c(price_candle)
                        logger.info(f"[REPLAY] Harga {p} di dalam FVG {fvg['low']}–{fvg['high']} — immediate trigger")
                        return {"event": "fvg_touched", "data": {
                            "fvg": fvg, "bos": self.bos,
                            "remaining_fvgs": self.fvgs,
                            "candle_touch": {"h": p, "l": p, "c": p, "ts": "live"},
                            "immediate": True,
                        }}

            for i in range(self.last_h1_idx, n):
                c = candles[i]

                # Cek CHOCH: harga melewati swing referensi (invalidasi BOS)
                if bos_dir == "bearish" and _c(c) > self.bos.get("sh_before_bos", 999999):
                    self.reset()
                    logger.info(f"[REPLAY] CHOCH — BOS invalid. Reset.")
                    return {"event": "choch", "data": {"reason": "close above sh_before_bos"}}

                if bos_dir == "bullish" and _c(c) < self.bos.get("sl_before_bos", 0):
                    self.reset()
                    return {"event": "choch", "data": {"reason": "close below sl_before_bos"}}

                # Cek FVG filled (kalau ada candle yang close menembus FVG, hapus FVG itu)
                self.fvgs = [f for f in self.fvgs if not check_fvg_filled(f, c)]

                if not self.fvgs:
                    # Semua FVG sudah filled — reset
                    logger.info("[REPLAY] Semua FVG filled → reset SCAN_BOS")
                    self.reset()
                    return {"event": "all_fvg_filled", "data": {}}

                # Cek apakah ada FVG yang disentuh (wick touch)
                for fvg in self.fvgs:
                    if check_fvg_touched(fvg, c):
                        touched_fvg = fvg
                        self.last_h1_idx = i + 1
                        self.state = "IDM_HUNT"
                        logger.info(f"[REPLAY] FVG {fvg['type']} disentuh @ {fvg['low']}–{fvg['high']}")
                        return {"event": "fvg_touched", "data": {
                            "fvg": touched_fvg, "bos": self.bos,
                            "remaining_fvgs": self.fvgs,
                            "candle_touch": {"h": _h(c), "l": _l(c), "c": _c(c), "ts": _ts(c)},
                        }}

            self.last_h1_idx = n
            return {"event": "none", "data": {"waiting": "fvg_touch",
                                               "fvgs": len(self.fvgs),
                                               "bos": self.bos}}

        return {"event": "none", "data": {}}

    def replay_m5(self, m5_candles: list, direction: str,
                  logic: dict, sh: float, sl: float) -> dict:
        """
        Replay M5 kiri ke kanan cari IDM dalam range SH-SL.
        Return: {"event": "idm_found"|"none", "data": {...}}
        """
        idm_cfg  = logic.get("find_idm_m5", {})
        dir_cfg  = idm_cfg.get(direction, {})
        gap_min  = dir_cfg.get("gap_min_candles", 1)
        max_srch = idm_cfg.get("max_search_candles", 200)

        # Filter candle dalam range SH-SL
        in_range = [c for c in m5_candles if sl * 0.995 <= _l(c) and _h(c) <= sh * 1.005]
        use = in_range if len(in_range) >= 5 else m5_candles

        idm = find_idm_replay(use, direction, gap_min, max_srch)
        if idm:
            # Validasi level dalam range
            if sl <= idm["watch_level"] <= sh:
                self.idm_m5    = idm
                self.idm_watch = idm["watch_level"]
                self.state     = "BOS_GUARD"
                logger.info(f"[REPLAY M5] IDM {direction} @ watch={idm['watch_level']}")
                return {"event": "idm_found", "data": idm}

        return {"event": "none", "data": {"searching": direction, "range": f"{sl}–{sh}"}}

    def check_idm_touched(self, candle: dict) -> bool:
        """Cek apakah candle baru menyentuh level IDM."""
        if not self.idm_watch:
            return False
        idm = self.idm_m5
        if not idm:
            return False
        if idm["type"] == "bearish_idm":
            return _l(candle) <= self.idm_watch or _c(candle) <= self.idm_watch
        else:
            return _h(candle) >= self.idm_watch or _c(candle) >= self.idm_watch

    def reset(self):
        """Reset ke SCAN_BOS dari posisi saat ini (tidak dari awal)."""
        self.state       = "SCAN_BOS"
        self.bos         = None
        self.fvgs        = []
        self.idm_m5      = None
        self.idm_watch   = 0.0
        # Simpan swing_highs/lows dan last_h1_idx — tidak perlu replay ulang dari awal

    def to_dict(self) -> dict:
        return {
            "state": self.state, "bos": self.bos,
            "fvgs": self.fvgs, "fvgs_count": len(self.fvgs),
            "idm_m5": self.idm_m5, "idm_watch": self.idm_watch,
            "last_h1_idx": self.last_h1_idx,
            "swing_highs_count": len(self.swing_highs),
            "swing_lows_count": len(self.swing_lows),
        }


# ── Helper untuk format hasil replay ke teks AI ─────────

def format_replay_for_ai(replay_event: dict, engine: "ReplayEngine") -> str:
    """Format hasil replay jadi teks ringkas untuk AI."""
    event = replay_event.get("event", "none")
    data  = replay_event.get("data", {})
    lines = []

    if event == "bos":
        bos = data.get("bos", {})
        fvgs = data.get("fvgs", [])
        lines.append(f"=== BOS {bos.get('type','').upper()} TERBENTUK ===")
        lines.append(f"Swing level: {bos.get('swing_level')} (idx={bos.get('swing_idx')})")
        lines.append(f"Dikonfirmasi: idx={bos.get('bos_candle_idx')}, close={bos.get('bos_close')}")
        lines.append(f"SH sebelum BOS: {bos.get('sh_before_bos') or bos.get('sh_since_bos')}")
        lines.append(f"SL referensi: {bos.get('sl_before_bos') or bos.get('sl_since_bos')}")
        lines.append(f"\nFVG WATCHLIST ({len(fvgs)}):")
        for f in fvgs:
            lines.append(f"  {f['type'].upper()} FVG: {f['low']}–{f['high']} gap={f['gap_pct']:.3f}%")

    elif event == "fvg_touched":
        fvg  = data.get("fvg", {})
        bos  = data.get("bos", {})
        ct   = data.get("candle_touch", {})
        lines.append(f"=== FVG {fvg.get('type','').upper()} DISENTUH ===")
        lines.append(f"FVG zone: {fvg['low']}–{fvg['high']}")
        lines.append(f"Candle touch: H={ct.get('h')} L={ct.get('l')} C={ct.get('c')}")
        lines.append(f"BOS aktif: {bos.get('type')} @ {bos.get('swing_level')}")
        lines.append(f"SH={bos.get('sh_before_bos') or bos.get('sh_since_bos')} "
                     f"SL={bos.get('sl_before_bos') or bos.get('sl_since_bos')}")
        lines.append(f"→ Sekarang: cari IDM M5 dalam range SH-SL")

    elif event == "idm_found":
        idm = data
        lines.append(f"=== IDM {idm.get('type','').upper()} DITEMUKAN ===")
        lines.append(f"Candle A: idx={idm.get('candle_a_idx')} H={idm.get('candle_a_high')} L={idm.get('candle_a_low')}")
        lines.append(f"Watch level: {idm.get('watch_level')} (harga harus sentuh ini)")
        lines.append(f"Gap: {idm.get('gap')} candle")

    elif event == "none":
        state = engine.state if engine else "unknown"
        if state == "SCAN_BOS":
            lines.append("Status: scanning BOS H1...")
        elif state == "WAIT_FVG":
            fvgs = engine.fvgs if engine else []
            lines.append(f"Status: menunggu FVG touch | {len(fvgs)} FVG aktif")
            for f in fvgs:
                lines.append(f"  {f['type']} {f['low']}–{f['high']}")

    return "\n".join(lines) if lines else "Status: monitoring..."
