"""
Specialist Agents — 4 AI dengan tugas terbagi, token-efficient.

Prinsip:
- Setiap AI terima data MINIMAL yang relevan dengan tugasnya saja
- Tidak ada AI yang handle semua sekaligus
- Diskusi hanya saat loss, bukan setiap siklus
- Semua pakai watchlist trigger — tidak mantau terus menerus

Alur:
  AI-1 (M15 Analyst)   → baca struktur M15, FVG, OB → set watchlist level M15
  AI-2 (IDM Hunter)    → baca M1, cari IDM → set notif di high/low IDM
  AI-3 (BOS/MSS Guard) → saat IDM disentuh → konfirmasi BOS lanjut atau MSS
  AI-4 (Entry Sniper)  → saat BOS terkonfirmasi → entry presisi, SL MSNR, TP target
"""

import os
import json
import logging
from datetime import datetime, timezone
from groq import Groq

logger = logging.getLogger(__name__)


def _parse_json(raw: str) -> dict | list | None:
    """Parse JSON dari response model, handle berbagai format."""
    import re
    if not raw:
        return None
    raw = raw.strip()
    try:
        return json.loads(raw)
    except Exception:
        pass
    match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```', raw)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    match = re.search(r'(\{[\s\S]*\})', raw)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    match = re.search(r'(\[[\s\S]*\])', raw)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass
    return None


def _call(client: Groq, model: str, prompt: str, max_tokens: int = 400, temp: float = 0.2) -> str:
    """Panggil Groq API, return teks mentah."""
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=max_tokens,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        err = str(e)
        if "429" in err or "rate_limit" in err.lower():
            logger.warning(f"[RATE LIMIT] {model}: {err[:100]}")
        elif "413" in err:
            logger.warning(f"[PAYLOAD TOO LARGE] {model} — prompt terlalu panjang")
        else:
            logger.error(f"[API ERROR] {model}: {err[:100]}")
        return ""


# ══════════════════════════════════════════════════════════
# AI-1: M15 ANALYST
# Tugas: deteksi BOS M15 → cari IDM M15 → pasang watchlist
# Alur:
#   1. Deteksi BOS M15 (Python sudah hitung)
#   2. Cari IDM M15 setelah BOS (Python sudah hitung)
#   3. Pasang watchlist di level IDM M15
#   4. Cek OB sweep fakeout sebelum serahkan ke AI-2
# Token: sangat ringan — hanya terima hasil Python, beri narasi
# ══════════════════════════════════════════════════════════

def ai1_m15_analysis(client: Groq, model: str, ict_data: dict, current_price: float,
                     bos_m15: dict = None, idm_m15: dict = None,
                     ob_sweep: dict = None) -> dict:
    """
    AI-1 narasi BOS + IDM M15, konfirmasi watchlist.
    Semua deteksi sudah dilakukan Python — AI hanya narasi dan konfirmasi.
    """
    # Jika tidak ada BOS — cukup report neutral
    if not bos_m15:
        bias = ict_data.get("m15_bias", {})
        return {
            "bias": bias.get("direction", "neutral"),
            "reason": bias.get("reason", "belum ada BOS"),
            "bos_found": False,
            "chat_msg": f"Hiura: belum ada BOS M15 yang bersih. Bias {bias.get('direction','neutral')}. Pantau dulu.",
            "watchlist": [],
        }

    bos_type = bos_m15.get("type", "")
    bos_level = bos_m15.get("level", 0)
    bias = "bullish" if bos_type == "bullish_bos" else "bearish"

    # Cek fakeout OB sweep
    fakeout_msg = ""
    if ob_sweep and ob_sweep.get("fakeout"):
        fakeout_msg = f"\n⚠️ FAKEOUT ALERT: {ob_sweep.get('message','')}"

    # Watchlist: pasang di level IDM M15 jika ada
    watchlist = []
    idm_level = 0

    if idm_m15 and not (ob_sweep and ob_sweep.get("fakeout")):
        # watch_level = low candle A (BOS bullish) atau high candle A (BOS bearish)
        idm_level = idm_m15.get("watch_level", idm_m15.get("level", 0))
        idm_type  = idm_m15.get("type", "")
        desc      = idm_m15.get("description", f"IDM M15 {idm_type} @ {idm_level:.2f}")
        if idm_level > 0:
            watchlist.append({
                "level": idm_level,
                "condition": "touch",
                "reason": desc,
                "for_ai": "AI-1-IDM-TOUCHED",
                "phase": "waiting_idm_touch",
            })

    # Jika fakeout — pasang watchlist konfirmasi OB break
    elif ob_sweep and ob_sweep.get("fakeout"):
        ob_lvl = ob_sweep.get("ob_level", 0)
        if bias == "bullish":
            watchlist.append({
                "level": ob_lvl,
                "condition": "break_below",
                "reason": f"Konfirmasi OB break — close M15 di bawah {ob_lvl:.2f} = bearish valid",
                "for_ai": "AI-1",
                "phase": "waiting_ob_confirm",
            })
        else:
            watchlist.append({
                "level": ob_lvl,
                "condition": "break_above",
                "reason": f"Konfirmasi OB break — close M15 di atas {ob_lvl:.2f} = bullish valid",
                "for_ai": "AI-1",
                "phase": "waiting_ob_confirm",
            })

    # Narasi singkat untuk chat grup
    if ob_sweep and ob_sweep.get("fakeout"):
        chat_msg = f"Hiura: ada BOS {bias} M15 di {bos_level:.2f}, tapi OB ke-sweep — kemungkinan fakeout. Tunggu konfirmasi close dulu.{fakeout_msg}"
    elif idm_level > 0:
        chat_msg = f"Hiura: BOS {bias} M15 terkonfirmasi di {bos_level:.2f}. Pantau low IDM M15 @ {idm_level:.2f} — retrace harus sentuh sini."
    else:
        chat_msg = f"Hiura: BOS {bias} M15 di {bos_level:.2f}, tapi IDM M15 belum terbentuk. Pantau dulu."

    logger.info(f"[AI-1] BOS={bos_type} @ {bos_level:.2f} | IDM={idm_level:.2f} | "
                f"fakeout={ob_sweep.get('fakeout') if ob_sweep else False} | watchlist={len(watchlist)}")

    return {
        "bias": bias,
        "bos_found": True,
        "bos_type": bos_type,
        "bos_level": bos_level,
        "idm_m15_level": idm_level,
        "reason": f"BOS {bias} M15 @ {bos_level:.2f}",
        "chat_msg": chat_msg,
        "watchlist": watchlist,
        "fakeout_detected": bool(ob_sweep and ob_sweep.get("fakeout")),
    }


# ══════════════════════════════════════════════════════════
# AI-2: IDM HUNTER (M1)
# Dipanggil setelah IDM M15 disentuh & swing range sudah ditandai AI-1
# BOS bullish M15 → cari IDM BEARISH di M1 (retrace turun dalam range SH-SL)
# BOS bearish M15 → cari IDM BULLISH di M1 (retrace naik dalam range SH-SL)
# Token: sangat ringan — Python sudah hitung IDM, AI hanya narasi + fallback
# ══════════════════════════════════════════════════════════

def ai2_idm_hunter(client: Groq, model: str, ict_data: dict, idm_m1: dict | None,
                   swing_range: dict, current_price: float) -> dict:
    """
    AI-2 set watchlist IDM M1 dalam range SH-SL dari AI-1.
    idm_m1: hasil Python find_idm_in_range() — sudah presisi
    swing_range: {"swing_high", "swing_low", "m1_idm_direction", "range_valid"}
    """
    sh  = swing_range.get("swing_high", 0)
    sl  = swing_range.get("swing_low", 0)
    m1_dir = swing_range.get("m1_idm_direction", "bearish")
    range_valid = swing_range.get("range_valid", False)
    watchlist = []

    if idm_m1 and range_valid:
        idm_level = idm_m1.get("level", 0)
        idm_type  = idm_m1.get("type", "")

        # IDM bearish M1 (BOS bullish M15): harga retrace turun sentuh level ini
        # IDM bullish M1 (BOS bearish M15): harga retrace naik sentuh level ini
        watchlist.append({
            "level": idm_level,
            "condition": "touch",
            "reason": f"IDM {idm_type} M1 @ {idm_level:.2f} dalam range {sl:.2f}–{sh:.2f}",
            "for_ai": "AI-3",
            "phase": "waiting_idm_m1_touch",
        })
        chat_msg = (
            f"Senanan: IDM {m1_dir} M1 ketemu di {idm_level:.2f} "
            f"(range {sl:.2f}–{sh:.2f}). Pantau level ini."
        )
        logger.info(f"[AI-2] IDM M1 {m1_dir} @ {idm_level:.2f} | range {sl:.2f}–{sh:.2f}")
        return {
            "idm_valid": True,
            "idm_level": idm_level,
            "idm_direction": m1_dir,
            "chat_msg": chat_msg,
            "watchlist": watchlist,
        }

    # IDM M1 belum terbentuk dalam range — minta AI konfirmasi swing
    # dan set watchlist sementara di SL/SH range sebagai level pantau
    if range_valid:
        # Pantau batas range: kalau break SL (bullish) atau SH (bearish) = invalidasi
        if m1_dir == "bearish":
            # BOS bullish M15 — kalau SL (swing_low) break ke bawah = invalidasi
            watchlist.append({
                "level": sl,
                "condition": "break_below",
                "reason": f"Batas bawah range — break SL {sl:.2f} = invalidasi BOS bullish M15",
                "for_ai": "AI-1",
                "phase": "waiting_idm_m1_touch",
            })
        else:
            watchlist.append({
                "level": sh,
                "condition": "break_above",
                "reason": f"Batas atas range — break SH {sh:.2f} = invalidasi BOS bearish M15",
                "for_ai": "AI-1",
                "phase": "waiting_idm_m1_touch",
            })
        chat_msg = (
            f"Senanan: IDM {m1_dir} M1 belum terbentuk di range {sl:.2f}–{sh:.2f}. "
            f"Pantau batas range dulu."
        )
        logger.info(f"[AI-2] IDM M1 belum ada | range {sl:.2f}–{sh:.2f} | pantau batas")
    else:
        chat_msg = "Senanan: range dari AI-1 belum valid, tunggu IDM M15 disentuh dulu."
        logger.warning("[AI-2] swing_range tidak valid")

    return {
        "idm_valid": False,
        "idm_level": 0,
        "idm_direction": m1_dir,
        "chat_msg": chat_msg,
        "watchlist": watchlist,
    }


# ══════════════════════════════════════════════════════════
# AI-3: BOS/MSS GUARD
# Tugas: setelah IDM disentuh → konfirmasi BOS lanjut atau MSS
# Output: keputusan BOS/MSS + instruksi ke AI-2 atau AI-4
# Token: ringan — hanya konteks IDM + candle terbaru M1
# ══════════════════════════════════════════════════════════

def ai3_bos_mss_guard(client: Groq, model: str, ict_data: dict,
                      bos_result: dict | None, idm_level: float,
                      bias: str, current_price: float) -> dict:
    """
    AI-3 konfirmasi BOS atau MSS setelah IDM disentuh.
    Return: {"decision": "bos|mss|wait", "chat_msg": str, "next": "AI-2|AI-4", "watchlist": [...]}
    """
    mss = ict_data.get("m1_mss")

    bos_text = "Tidak ada BOS M1 terdeteksi"
    if bos_result:
        bos_text = f"BOS M1 {bos_result['type']}: level={bos_result['level']:.2f} close={bos_result['close']:.2f}"

    mss_text = "Tidak ada MSS terdeteksi"
    if mss:
        mss_text = f"MSS {mss.get('type','')}: level={mss.get('level',0):.2f}"

    prompt = f"""Kamu adalah AI-3, penjaga BOS/MSS di M1. Singkat dan tegas.

KONTEKS:
Bias M15: {bias}
IDM yang baru disentuh: {idm_level:.2f}
Harga saat ini: {current_price:.2f}

STATUS M1:
BOS M1: {bos_text}
MSS M1: {mss_text}

ATURAN KEPUTUSAN:
- Jika ada BOS M1 searah dengan bias M15 → keputusan = "bos" → lanjut ke AI-4
- Jika ada MSS (BOS berlawanan arah) → keputusan = "mss" → arahkan AI-2 cari IDM baru
- Jika belum ada BOS maupun MSS → keputusan = "wait" → set watchlist untuk konfirmasi

WAJIB: balas HANYA JSON murni:
{{
  "decision": "bos|mss|wait",
  "reason": "1 kalimat alasan keputusan",
  "chat_msg": "pesan ke grup (1-2 kalimat, gaya WA — kalau MSS bilang ke AI-2 untuk cari IDM baru)",
  "next": "AI-2|AI-4|wait",
  "watchlist": [
    {{"level": 0.0, "condition": "break_above|break_below", "reason": "BOS confirmation level", "for_ai": "AI-3|AI-4"}}
  ]
}}"""

    raw = _call(client, model, prompt, max_tokens=280)
    parsed = _parse_json(raw)
    if not parsed:
        logger.warning(f"[AI-3] Gagal parse: {raw[:100]}")
        return {"decision": "wait", "reason": "parse error", "chat_msg": "", "next": "wait", "watchlist": []}

    logger.info(f"[AI-3] decision={parsed.get('decision')} | next={parsed.get('next')}")
    return parsed


# ══════════════════════════════════════════════════════════
# AI-4: ENTRY SNIPER
# Tugas: cari entry presisi setelah BOS terkonfirmasi
# Output: entry price, SL (MSNR close), TP target
# Belajar dari memory trade historis
# ══════════════════════════════════════════════════════════

def ai4_entry_sniper(client: Groq, model: str, ict_data: dict,
                     msnr_levels: list, current_price: float,
                     bias: str, bos_level: float,
                     trade_memory: list) -> dict:
    """
    AI-4 tentukan entry presisi, SL MSNR, TP.
    Return: {"entry": float, "sl": float, "tp": float, "setup_type": str, "chat_msg": str, "confidence": float}
    """
    # Format memory trade (max 5 trade relevan)
    mem_text = "Belum ada trade historis."
    if trade_memory:
        recent = trade_memory[-5:]
        mem_text = "\n".join([
            f"  [{t.get('result','?')}] {t.get('direction','?')} entry={t.get('entry',0):.2f} "
            f"setup={t.get('setup','?')} notes={t.get('notes','')[:60]}"
            for t in recent
        ])

    # MSNR levels
    msnr_text = "\n".join([
        f"  {m['type']}: {m['level']:.2f} ({m['source']})"
        for m in msnr_levels[:4]
    ]) or "  Tidak ada MSNR terdeteksi"

    # FVG M1 untuk Quasimodo / retest area
    fvgs_m1 = ict_data.get("m1_fvg", [])[:3]
    fvg_text = "\n".join([
        f"  {f['type'].upper()} FVG: {f['low']:.2f}–{f['high']:.2f} filled={f['filled']}"
        for f in fvgs_m1
    ]) or "  Tidak ada FVG M1"

    ob_m15 = ict_data.get("m15_ob", [])[:2]
    ob_text = "\n".join([
        f"  {ob['type'].upper()} OB M15: {ob['low']:.2f}–{ob['high']:.2f}"
        for ob in ob_m15
    ]) or "  Tidak ada OB M15"

    prompt = f"""Kamu adalah AI-4, spesialis entry presisi. Tidak pakai indikator — hanya candlestick.

KONTEKS:
Bias M15: {bias}
BOS terkonfirmasi di: {bos_level:.2f}
Harga saat ini: {current_price:.2f}

LEVEL MSNR (Malaysian S/R — hanya close candle, wick diabaikan):
{msnr_text}

FVG M1 (potensi Quasimodo / retest):
{fvg_text}

OB M15:
{ob_text}

HISTORIS TRADE (belajar dari sini):
{mem_text}

TUGASMU:
Tentukan entry PALING PRESISI berdasarkan setup ini. Gunakan angka PERSIS dari data.

Pilihan setup entry (berdasarkan konteks):
- RBS (Resistance Become Support): kalau harga retest level yang sebelumnya resistance
- Quasimodo: kalau ada FVG M1 yang belum terisi dekat area BOS
- OB retest: kalau harga pullback ke OB M15
- MSNR support: kalau ada level close candle yang bersih

SL: WAJIB di bawah level MSNR support (close candle, bukan wick) — gunakan harga PERSIS
TP: ke liquidity pool berikutnya (swing high terakhir M15 atau recent_high)

WAJIB: balas HANYA JSON murni:
{{
  "setup_type": "RBS|Quasimodo|OB_retest|MSNR_support",
  "entry": 0.0,
  "sl": 0.0,
  "tp": 0.0,
  "rr": 0.0,
  "sl_reason": "level MSNR close yang jadi acuan SL",
  "tp_reason": "target liquidity yang dituju",
  "chat_msg": "pesan ke grup: setup apa, entry di mana, SL/TP (1-3 kalimat, gaya WA)",
  "confidence": 0.0
}}"""

    raw = _call(client, model, prompt, max_tokens=400, temp=0.15)
    parsed = _parse_json(raw)
    if not parsed:
        logger.warning(f"[AI-4] Gagal parse: {raw[:100]}")
        return {"setup_type": "unknown", "entry": current_price, "sl": 0, "tp": 0,
                "rr": 0, "chat_msg": "", "confidence": 0}

    logger.info(f"[AI-4] setup={parsed.get('setup_type')} | entry={parsed.get('entry')} | "
                f"sl={parsed.get('sl')} | tp={parsed.get('tp')} | conf={parsed.get('confidence')}")
    return parsed


# ══════════════════════════════════════════════════════════
# LOSS DEBRIEF — Diskusi hanya saat loss
# Analisis apa yang salah dari 4 AI
# ══════════════════════════════════════════════════════════

def loss_debrief(clients: list, models: list, closed_trade: dict, trade_context: dict) -> dict:
    """
    Panggil semua 4 AI untuk evaluasi trade yang loss.
    Return: {"ai1_verdict": str, "ai2_verdict": str, "ai3_verdict": str, "ai4_verdict": str,
             "root_cause": str, "lesson": str, "chat_log": list}
    """
    chat_log = []

    trade_summary = (
        f"Trade {closed_trade.get('direction','?').upper()} | "
        f"Entry: {closed_trade.get('entry',0):.2f} | "
        f"SL: {closed_trade.get('sl',0):.2f} | Exit: {closed_trade.get('exit_price',0):.2f} | "
        f"PnL: {closed_trade.get('pnl',0):.4f} | Setup: {closed_trade.get('setup','?')}"
    )

    # Masing-masing AI evaluasi bagiannya
    ai_names = ["AI-1 (M15)", "AI-2 (IDM)", "AI-3 (BOS/MSS)", "AI-4 (Entry)"]
    ai_roles = [
        "Apakah bias M15 sudah benar? Apakah OB/FVG yang digunakan valid?",
        "Apakah IDM yang dipilih valid? Apakah level IDM sudah tepat?",
        "Apakah keputusan BOS vs MSS sudah benar? Apakah seharusnya MSS tapi dilanjutkan?",
        "Apakah entry, SL, dan TP sudah di level yang tepat? Apakah setup yang dipilih sesuai konteks?"
    ]

    verdicts = {}
    for i, (client, model, name, role) in enumerate(zip(clients, models, ai_names, ai_roles)):
        prompt = f"""Kamu adalah {name}. Ada trade yang loss — evaluasi bagianmu.

TRADE YANG LOSS:
{trade_summary}

PERTANYAAN UNTUKMU:
{role}

Berikan evaluasi singkat 2-3 kalimat. Jujur, tidak perlu defensif.
Format: "{name}: [evaluasimu]"
Tulis HANYA pesanmu (plain text)."""

        raw = _call(client, model, prompt, max_tokens=200, temp=0.4)
        key = f"ai{i+1}_verdict"
        verdicts[key] = raw
        chat_log.append({"ai": f"ai{i+1}", "nama": name, "pesan": raw, "ronde": 1})
        logger.info(f"[DEBRIEF {name}] {raw[:100]}")

    # Kesimpulan root cause (pakai AI-3 yang paling sering salah)
    debrief_text = "\n".join([f"{k}: {v}" for k, v in verdicts.items()])
    prompt_rc = f"""Berdasarkan evaluasi 4 AI berikut setelah trade loss:

{debrief_text}

Trade: {trade_summary}

Identifikasi:
1. AI mana yang paling bertanggung jawab atas loss ini?
2. Apa root cause utamanya?
3. Apa 1 rule yang harus ditambahkan untuk mencegah hal ini terulang?

WAJIB: balas HANYA JSON murni:
{{
  "culprit": "AI-1|AI-2|AI-3|AI-4|kombinasi",
  "root_cause": "penjelasan singkat",
  "new_rule": "rule baru yang harus diterapkan",
  "lesson": "1 kalimat pelajaran"
}}"""

    raw_rc = _call(clients[2], models[2], prompt_rc, max_tokens=250, temp=0.3)
    rc = _parse_json(raw_rc) or {
        "culprit": "unknown", "root_cause": "parse error",
        "new_rule": "", "lesson": "review manual diperlukan"
    }

    chat_log.append({
        "ai": "system",
        "nama": "📋 Debrief",
        "pesan": f"Root cause: {rc.get('culprit')} — {rc.get('root_cause')} | Rule baru: {rc.get('new_rule')}",
        "ronde": 2
    })

    return {**verdicts, **rc, "chat_log": chat_log}
