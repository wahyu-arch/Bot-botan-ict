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
# Tugas: baca struktur M15, FVG, OB, tentukan bias
# Output: bias + level watchlist M15 yang perlu dipantau
# Token: ringan — hanya data M15 ringkas
# ══════════════════════════════════════════════════════════

def ai1_m15_analysis(client: Groq, model: str, ict_data: dict, current_price: float) -> dict:
    """
    AI-1 analisis struktur M15.
    Return: {"bias": "bullish|bearish|neutral", "reason": "...", "watchlist": [...], "chat_msg": "..."}
    """
    bias = ict_data.get("m15_bias", {})
    obs  = ict_data.get("m15_ob", [])[:3]   # max 3 OB
    fvgs = ict_data.get("m15_fvg", [])[:3]  # max 3 FVG
    liq  = ict_data.get("liquidity_pools", {})

    # Format ringkas
    ob_text = "\n".join([
        f"  {ob['type'].upper()} OB: {ob['low']:.2f}–{ob['high']:.2f} | mitigated={ob['mitigation']}"
        for ob in obs
    ]) or "  Tidak ada OB terdeteksi"

    fvg_text = "\n".join([
        f"  {f['type'].upper()} FVG: {f['low']:.2f}–{f['high']:.2f} (mid={f['midpoint']:.2f}) filled={f['filled']}"
        for f in fvgs
    ]) or "  Tidak ada FVG terdeteksi"

    prompt = f"""Kamu adalah AI-1, spesialis struktur M15. Analisis singkat dan presisi.

DATA M15:
Bias saat ini: {bias.get('direction','?')} — {bias.get('reason','')}
Swing High: {bias.get('last_swing_high', 0):.2f} | Swing Low: {bias.get('last_swing_low', 0):.2f}

ORDER BLOCKS:
{ob_text}

FAIR VALUE GAPS:
{fvg_text}

LIQUIDITY: High={liq.get('recent_high',0):.2f} | Low={liq.get('recent_low',0):.2f}
Harga saat ini: {current_price:.2f}

TUGASMU:
1. Konfirmasi bias M15 (bullish/bearish/neutral) dengan alasan 1 kalimat
2. Tentukan 2-3 level M15 yang HARUS dipantau sebagai trigger (gunakan angka PERSIS dari data)
   - Untuk bullish: OB low atau FVG low yang belum terisi sebagai area support
   - Untuk bearish: OB high atau FVG high yang belum terisi sebagai area resistance

WAJIB: balas HANYA JSON murni, langsung dari {{ :
{{
  "bias": "bullish|bearish|neutral",
  "reason": "1 kalimat alasan bias",
  "chat_msg": "pesan singkat ke grup (1-2 kalimat, gaya WA)",
  "watchlist": [
    {{"level": 71787.45, "condition": "touch|break_above|break_below", "reason": "OB bullish low M15", "for_ai": "AI-2"}}
  ]
}}"""

    raw = _call(client, model, prompt, max_tokens=350)
    parsed = _parse_json(raw)
    if not parsed:
        logger.warning(f"[AI-1] Gagal parse response: {raw[:100]}")
        return {"bias": "neutral", "reason": "parse error", "chat_msg": "", "watchlist": []}

    logger.info(f"[AI-1] bias={parsed.get('bias')} | {len(parsed.get('watchlist',[]))} level set")
    return parsed


# ══════════════════════════════════════════════════════════
# AI-2: IDM HUNTER
# Tugas: cari IDM terbaru di M1 sesuai bias M15
# Output: level IDM yang harus disentuh + notif watchlist
# Token: ringan — hanya data M1 ringkas + IDM hasil Python
# ══════════════════════════════════════════════════════════

def ai2_idm_hunter(client: Groq, model: str, ict_data: dict, idm_result: dict | None,
                   bias: str, current_price: float) -> dict:
    """
    AI-2 cari dan konfirmasi IDM di M1.
    Return: {"idm_level": float, "direction": str, "chat_msg": str, "watchlist": [...]}
    """
    m1_fvgs = ict_data.get("m1_fvg", [])[:2]
    fvg_text = "\n".join([
        f"  {f['type'].upper()} FVG M1: {f['low']:.2f}–{f['high']:.2f} filled={f['filled']}"
        for f in m1_fvgs
    ]) or "  Tidak ada FVG M1"

    idm_text = "Tidak ada IDM terdeteksi oleh sistem"
    if idm_result:
        idm_text = (
            f"IDM {idm_result['type']} ditemukan\n"
            f"  Level (high/low candle A): {idm_result['level']:.2f}\n"
            f"  Candle A time: {idm_result.get('candle_a_time','')}\n"
            f"  Tembus time: {idm_result.get('tembus_time','')}"
        )

    prompt = f"""Kamu adalah AI-2, spesialis IDM (Inducement) di M1. Singkat dan presisi.

BIAS M15: {bias}
HARGA SAAT INI: {current_price:.2f}

IDM TERDETEKSI SISTEM:
{idm_text}

FVG M1:
{fvg_text}

TUGASMU:
{"IDM sudah terdeteksi sistem. Konfirmasi level IDM ini valid untuk bias " + bias + ". Set watchlist: untuk bullish IDM → notif saat harga TURUN ke level IDM (touch/break_below). Untuk bearish IDM → notif saat harga NAIK ke level IDM (touch/break_above)." if idm_result else "IDM belum terdeteksi. Set watchlist di swing high/low M1 terdekat sebagai level pantauan IDM potensial."}

WAJIB: balas HANYA JSON murni:
{{
  "idm_valid": true,
  "idm_level": {idm_result['level'] if idm_result else 0},
  "idm_direction": "{bias}",
  "chat_msg": "pesan singkat ke grup (1-2 kalimat, gaya WA)",
  "watchlist": [
    {{"level": 0.0, "condition": "touch|break_above|break_below", "reason": "IDM level M1", "for_ai": "AI-3"}}
  ]
}}"""

    raw = _call(client, model, prompt, max_tokens=300)
    parsed = _parse_json(raw)
    if not parsed:
        logger.warning(f"[AI-2] Gagal parse: {raw[:100]}")
        # Fallback: gunakan IDM dari Python analyzer
        if idm_result:
            cond = "touch" if bias == "bullish" else "touch"
            return {
                "idm_valid": True,
                "idm_level": idm_result["level"],
                "idm_direction": bias,
                "chat_msg": f"IDM {idm_result['type']} di {idm_result['level']:.2f} — pantau level ini.",
                "watchlist": [{
                    "level": idm_result["level"],
                    "condition": "touch",
                    "reason": f"IDM {idm_result['type']} level",
                    "for_ai": "AI-3"
                }]
            }
        return {"idm_valid": False, "idm_level": 0, "chat_msg": "", "watchlist": []}

    logger.info(f"[AI-2] IDM level={parsed.get('idm_level')} | valid={parsed.get('idm_valid')}")
    return parsed


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
