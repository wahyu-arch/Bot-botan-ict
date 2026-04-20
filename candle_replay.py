"""
CandleReplay — Python replay candle kiri ke kanan, hasilnya dikirim ke AI.

AI tidak perlu baca semua candle mentah — Python sudah deteksi:
- Swing high/low valid (swing_left + swing_right)
- IDM state machine
- BOS/MSS
- FVG

AI menerima HASIL replay, bukan raw candle.
AI tugasnya: interpretasi + keputusan + watchlist.
"""

import logging
logger = logging.getLogger(__name__)


def find_swings(candles: list, swing_left: int = 8, swing_right: int = 8) -> dict:
    """
    Replay kiri ke kanan, temukan swing high/low valid.
    Swing valid = N candle kiri tidak lebih tinggi/rendah, DAN N candle kanan juga.
    Return: {"swing_highs": [(idx, price)], "swing_lows": [(idx, price)]}
    """
    n = len(candles)
    swing_highs = []
    swing_lows  = []

    for i in range(swing_left, n - swing_right):
        c = candles[i]

        # Cek swing high
        is_sh = all(candles[j]["high"] < c["high"] for j in range(i - swing_left, i)) and \
                all(candles[j]["high"] < c["high"] for j in range(i + 1, i + swing_right + 1))
        if is_sh:
            swing_highs.append((i, c["high"], c.get("ts", "")))

        # Cek swing low
        is_sl = all(candles[j]["low"] > c["low"] for j in range(i - swing_left, i)) and \
                all(candles[j]["low"] > c["low"] for j in range(i + 1, i + swing_right + 1))
        if is_sl:
            swing_lows.append((i, c["low"], c.get("ts", "")))

    return {"swing_highs": swing_highs, "swing_lows": swing_lows}


def find_bos(candles: list, swing_left: int = 8, swing_right: int = 8) -> dict | None:
    """
    Replay kiri ke kanan — temukan BOS terbaru.
    BOS = close candle melewati swing high/low valid terakhir.
    Return dict BOS atau None.
    """
    swings = find_swings(candles, swing_left, swing_right)
    sh_list = swings["swing_highs"]
    sl_list = swings["swing_lows"]

    if not sh_list and not sl_list:
        return None

    # Replay kiri ke kanan — cari BOS dari candle ke-N setelah swing terbentuk
    # BOS bearish: close di bawah swing low valid terakhir
    # BOS bullish: close di atas swing high valid terakhir

    last_bos = None

    for i in range(swing_left + swing_right + 1, len(candles)):
        c = candles[i]

        # Swing low yang terbentuk sebelum candle i (dengan swing_right sudah terpenuhi)
        valid_sl = [(idx, price, ts) for idx, price, ts in sl_list
                    if idx + swing_right < i]
        valid_sh = [(idx, price, ts) for idx, price, ts in sh_list
                    if idx + swing_right < i]

        if valid_sl:
            last_sl_idx, last_sl_price, last_sl_ts = valid_sl[-1]
            if c["close"] < last_sl_price:
                # BOS bearish terkonfirmasi
                # SH = high tertinggi setelah BOS candle sebelumnya
                post_sh = [candles[j]["high"] for j in range(last_sl_idx, i + 1)]
                sh_since_bos = max(post_sh) if post_sh else c["high"]

                # SH sebelum swing low (untuk invalidasi)
                pre_sh = [(idx, p) for idx, p, _ in valid_sh if idx < last_sl_idx]
                sh_before_bos = pre_sh[-1][1] if pre_sh else c["high"]

                last_bos = {
                    "type": "bearish_bos",
                    "direction": "bearish",
                    "swing_level": round(last_sl_price, 6),
                    "swing_idx": last_sl_idx,
                    "swing_ts": last_sl_ts,
                    "bos_candle_idx": i,
                    "bos_close": round(c["close"], 6),
                    "bos_ts": c.get("ts", ""),
                    "sl_since_bos": round(min(candles[j]["low"] for j in range(last_sl_idx, i+1)), 6),
                    "sh_before_bos": round(sh_before_bos, 6),
                    "sh_since_bos": round(sh_since_bos, 6),
                }

        if valid_sh:
            last_sh_idx, last_sh_price, last_sh_ts = valid_sh[-1]
            if c["close"] > last_sh_price:
                # BOS bullish terkonfirmasi
                post_sl = [candles[j]["low"] for j in range(last_sh_idx, i + 1)]
                sl_since_bos = min(post_sl) if post_sl else c["low"]

                pre_sl = [(idx, p) for idx, p, _ in valid_sl if idx < last_sh_idx]
                sl_before_bos = pre_sl[-1][1] if pre_sl else c["low"]

                last_bos = {
                    "type": "bullish_bos",
                    "direction": "bullish",
                    "swing_level": round(last_sh_price, 6),
                    "swing_idx": last_sh_idx,
                    "swing_ts": last_sh_ts,
                    "bos_candle_idx": i,
                    "bos_close": round(c["close"], 6),
                    "bos_ts": c.get("ts", ""),
                    "sh_since_bos": round(max(candles[j]["high"] for j in range(last_sh_idx, i+1)), 6),
                    "sl_before_bos": round(sl_before_bos, 6),
                    "sl_since_bos": round(sl_since_bos, 6),
                }

    return last_bos


def find_fvg(candles: list, bos_type: str = "", min_gap_pct: float = 0.05) -> list:
    """
    Replay kiri ke kanan — temukan FVG setelah BOS.
    FVG bearish: candle[i].low > candle[i+2].high
    FVG bullish: candle[i].high < candle[i+2].low
    Hanya FVG yang sesuai arah BOS.
    """
    fvgs = []
    direction = "bearish" if "bearish" in bos_type else "bullish" if "bullish" in bos_type else ""

    for i in range(len(candles) - 2):
        c1, c2, c3 = candles[i], candles[i+1], candles[i+2]

        if direction in ("bearish", ""):
            # FVG bearish: gap turun — low C1 > high C3
            if c1["low"] > c3["high"]:
                gap = c1["low"] - c3["high"]
                mid = (c1["low"] + c3["high"]) / 2
                if (gap / mid * 100) >= min_gap_pct:
                    # Cek apakah sudah filled (candle setelahnya close menembus zona)
                    filled = any(
                        candles[j]["close"] > c1["low"]
                        for j in range(i+3, len(candles))
                    )
                    fvgs.append({
                        "type": "bearish",
                        "high": round(c1["low"], 6),   # atas FVG = low C1
                        "low":  round(c3["high"], 6),  # bawah FVG = high C3
                        "mid":  round(mid, 6),
                        "gap_pct": round(gap / mid * 100, 4),
                        "candle_idx": i,
                        "ts": c1.get("ts", ""),
                        "filled": filled,
                    })

        if direction in ("bullish", ""):
            # FVG bullish: gap naik — high C1 < low C3
            if c1["high"] < c3["low"]:
                gap = c3["low"] - c1["high"]
                mid = (c3["low"] + c1["high"]) / 2
                if (gap / mid * 100) >= min_gap_pct:
                    filled = any(
                        candles[j]["close"] < c1["high"]
                        for j in range(i+3, len(candles))
                    )
                    fvgs.append({
                        "type": "bullish",
                        "high": round(c3["low"], 6),   # atas FVG = low C3
                        "low":  round(c1["high"], 6),  # bawah FVG = high C1
                        "mid":  round(mid, 6),
                        "gap_pct": round(gap / mid * 100, 4),
                        "candle_idx": i,
                        "ts": c1.get("ts", ""),
                        "filled": filled,
                    })

    return fvgs


def find_idm_m5(candles: list, direction: str = "bearish",
                gap_min: int = 1, max_search: int = 200) -> dict | None:
    """
    State machine IDM replay kiri ke kanan.

    IDM bearish (untuk BOS bullish H1 — M5 retrace turun):
      Candle A buat swing HIGH
      Minimal gap_min candle tidak tembus high A (close maupun wick)
      Candle berikutnya TEMBUS high A
      Watch level = LOW candle A

    IDM bullish (untuk BOS bearish H1 — M5 retrace naik):
      Candle A buat swing LOW
      Gap candle tidak tembus low A
      Tembus → Watch level = HIGH candle A
    """
    n = min(len(candles), max_search)
    idms = []

    for i in range(1, n - gap_min - 1):
        c_a = candles[i]

        if direction == "bearish":
            high_a = c_a["high"]
            # Candle A harus lebih tinggi dari candle sebelumnya (swing high candidate)
            if high_a <= candles[i-1]["high"]:
                continue

            # Cari gap + tembus
            gap_count = 0
            tembus_idx = None
            for j in range(i+1, min(i+max_search, n)):
                if candles[j]["high"] >= high_a or candles[j]["close"] >= high_a:
                    if gap_count >= gap_min:
                        tembus_idx = j
                    break
                gap_count += 1

            if tembus_idx and gap_count >= gap_min:
                idms.append({
                    "type": "bearish_idm",
                    "candle_a_idx": i,
                    "candle_a_high": round(c_a["high"], 6),
                    "candle_a_low":  round(c_a["low"],  6),
                    "watch_level":   round(c_a["low"],  6),  # LOW candle A
                    "tembus_idx":    tembus_idx,
                    "gap_count":     gap_count,
                    "ts":            c_a.get("ts", ""),
                })

        elif direction == "bullish":
            low_a = c_a["low"]
            if low_a >= candles[i-1]["low"]:
                continue

            gap_count = 0
            tembus_idx = None
            for j in range(i+1, min(i+max_search, n)):
                if candles[j]["low"] <= low_a or candles[j]["close"] <= low_a:
                    if gap_count >= gap_min:
                        tembus_idx = j
                    break
                gap_count += 1

            if tembus_idx and gap_count >= gap_min:
                idms.append({
                    "type": "bullish_idm",
                    "candle_a_idx": i,
                    "candle_a_high": round(c_a["high"], 6),
                    "candle_a_low":  round(c_a["low"],  6),
                    "watch_level":   round(c_a["high"], 6),  # HIGH candle A
                    "tembus_idx":    tembus_idx,
                    "gap_count":     gap_count,
                    "ts":            c_a.get("ts", ""),
                })

    return idms[-1] if idms else None


def build_h1_analysis(candles: list, logic: dict) -> dict:
    """
    Replay H1 kiri ke kanan — hasilnya dikirim ke Hiura.
    Hiura tidak perlu baca candle mentah — sudah pre-processed.
    """
    bos_cfg = logic.get("find_bos_h1", {})
    fvg_cfg = logic.get("find_fvg_h1", {})

    sw_left  = bos_cfg.get("swing_left",  8)
    sw_right = bos_cfg.get("swing_right", 8)
    min_gap  = fvg_cfg.get("min_gap_pct", 0.05)

    bos  = find_bos(candles, sw_left, sw_right)
    fvgs = find_fvg(candles, bos.get("type", "") if bos else "", min_gap)

    # Hanya FVG setelah BOS dan belum filled
    if bos:
        bos_idx = bos.get("bos_candle_idx", 0)
        fvgs = [f for f in fvgs if f["candle_idx"] > bos_idx and not f["filled"]]

    return {
        "bos":  bos,
        "fvgs": fvgs[:3],  # max 3 FVG fresh
        "swings": find_swings(candles, sw_left, sw_right),
        "current_price": candles[-1]["close"] if candles else 0,
    }


def build_m5_analysis(candles: list, logic: dict,
                      idm_direction: str, sh: float, sl: float) -> dict:
    """
    Replay M5 kiri ke kanan — hasilnya dikirim ke Senanan.
    Cari IDM dalam range SH-SL.
    """
    idm_cfg = logic.get("find_idm_m5", {})
    dir_cfg  = idm_cfg.get(idm_direction, {})
    gap_min  = dir_cfg.get("gap_min_candles", 1)
    max_s    = idm_cfg.get("max_search_candles", 200)

    # Filter candle dalam range harga
    in_range = [c for c in candles if sl <= c["low"] and c["high"] <= sh * 1.005]
    use = in_range if len(in_range) >= 5 else candles

    idm = find_idm_m5(use, idm_direction, gap_min, max_s)

    return {
        "idm":           idm,
        "idm_direction": idm_direction,
        "range_sh":      sh,
        "range_sl":      sl,
    }


def format_replay_for_ai(h1_result: dict = None, m5_result: dict = None) -> str:
    """Format hasil replay jadi teks ringkas untuk AI."""
    lines = []

    if h1_result:
        bos = h1_result.get("bos")
        if bos:
            lines.append(f"=== BOS H1 TERDETEKSI ===")
            lines.append(f"Type: {bos['type']}")
            lines.append(f"Swing level (BOS di sini): {bos['swing_level']}")
            lines.append(f"Terbentuk di candle idx={bos['swing_idx']} ({bos['swing_ts']})")
            lines.append(f"Dikonfirmasi di candle idx={bos['bos_candle_idx']} ({bos['bos_ts']}), close={bos['bos_close']}")
            if bos["type"] == "bullish_bos":
                lines.append(f"SH sejak BOS: {bos.get('sh_since_bos',0)}")
                lines.append(f"SL sebelum BOS: {bos.get('sl_before_bos',0)}")
            else:
                lines.append(f"SL sejak BOS: {bos.get('sl_since_bos',0)}")
                lines.append(f"SH sebelum BOS: {bos.get('sh_before_bos',0)}")
        else:
            lines.append("=== TIDAK ADA BOS H1 ===")

        fvgs = h1_result.get("fvgs", [])
        if fvgs:
            lines.append(f"\n=== FVG H1 FRESH ({len(fvgs)}) ===")
            for f in fvgs:
                lines.append(f"  {f['type'].upper()} FVG: high={f['high']} low={f['low']} mid={f['mid']} gap={f['gap_pct']:.3f}%")
        else:
            lines.append("\n=== TIDAK ADA FVG H1 FRESH ===")

    if m5_result:
        idm = m5_result.get("idm")
        lines.append(f"\n=== IDM M5 ({m5_result['idm_direction'].upper()}) ===")
        if idm:
            lines.append(f"Type: {idm['type']}")
            lines.append(f"Candle A idx={idm['candle_a_idx']} ({idm['ts']})")
            lines.append(f"  High candle A: {idm['candle_a_high']}")
            lines.append(f"  Low candle A:  {idm['candle_a_low']}")
            lines.append(f"Watch level (harus disentuh): {idm['watch_level']}")
            lines.append(f"Gap count: {idm['gap_count']}")
        else:
            lines.append(f"IDM {m5_result['idm_direction']} tidak ditemukan dalam range {m5_result['range_sl']}–{m5_result['range_sh']}")

    return "\n".join(lines)
