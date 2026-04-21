"""
AI Analysts — 4 AI yang memutuskan semua analisis trading.

Python hanya kasih data mentah (candle OHLC).
AI yang memutuskan: ada BOS? FVG mana? IDM dimana? Entry di mana? SL/TP berapa?

Hiura  (AI-1) → Analisis H1: BOS + FVG + tentukan range SH/SL
Senanan (AI-2) → Analisis M5: IDM + set watchlist level
Shina  (AI-3) → Setelah IDM disentuh: BOS atau MSS M5?
Yusuf  (AI-4) → Entry presisi, SL, TP dari data yang terkumpul
"""

import json
import logging
import re
from groq import Groq
from candle_replay import ReplayEngine, format_replay_for_ai

logger = logging.getLogger(__name__)


def _load_json_files() -> dict:
    """Baca langsung dari file JSON di data/ — sumber kebenaran tunggal."""
    import os
    result = {}
    files = {
        "rules":  "data/rules.json",
        "logic":  "data/logic_rules.json",
        "prompts": "data/prompts.json",
    }
    for key, path in files.items():
        if os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as f:
                    raw = json.load(f)
                # Strip metadata keys
                result[key] = {k: v for k, v in raw.items() if not k.startswith("_")}
            except Exception as e:
                result[key] = {}
                logger.warning(f"[JSON] Gagal baca {path}: {e}")
        else:
            result[key] = {}
    result["logic_raw"] = result.get("logic", {})  # alias untuk akses field spesifik
    return result


def _build_json_ctx(ctx: dict) -> str:
    """Bangun konteks JSON lengkap untuk AI — rules, logic, dan prompts dari file JSON."""
    if not ctx:
        return ""
    parts = []
    if ctx.get("rules"):
        parts.append(f"=== RULES (parameter trading) ===\n{ctx['rules']}")
    if ctx.get("logic"):
        parts.append(f"=== LOGIC (cara deteksi BOS/FVG/IDM) ===\n{ctx['logic']}")
    if ctx.get("prompts"):
        parts.append(f"=== PROMPTS (instruksi khusus) ===\n{ctx['prompts']}")
    return "\n\n".join(parts) if parts else ""


# ── Helpers ─────────────────────────────────────────────

def _call(client: Groq, model: str, prompt: str,
          max_tokens: int = 500, temp: float = 0.3) -> str:
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
        tag = "[RATE LIMIT]" if ("429" in err or "rate_limit" in err.lower() or "413" in err) else "[API ERROR]"
        logger.warning(f"{tag} {model}: {err[:100]}")
        return ""


def _parse_json(raw: str) -> dict | list | None:
    if not raw:
        return None
    for attempt in [raw, re.search(r'```(?:json)?\s*([\s\S]*?)\s*```', raw),
                    re.search(r'(\{[\s\S]*\})', raw),
                    re.search(r'(\[[\s\S]*\])', raw)]:
        text = attempt.group(1) if hasattr(attempt, 'group') else attempt
        if not text:
            continue
        try:
            return json.loads(text)
        except Exception:
            pass
    return None


def _candle_table(candles: list, limit: int = 40) -> str:
    """Format candle OHLC jadi teks ringkas."""
    rows = []
    for c in candles[-limit:]:
        d = "↑" if c.get("bull") else "↓"
        rows.append(f"[{c['i']:3}] {c.get('ts','')[-5:]} {d} H:{c['h']} L:{c['l']} C:{c['c']}")
    return "\n".join(rows)


# ══════════════════════════════════════════════════════════
# HIURA (AI-1) — Analisis H1
# Input: candle H1 mentah
# Output: ada BOS? FVG mana? SH/SL berapa? Bias?
# ══════════════════════════════════════════════════════════

def hiura_h1_analysis(client: Groq, model: str, raw_data: dict,
                      ctx: dict = None, prompt_ctx: str = "") -> dict:
    """
    Hiura analisis H1 dari data mentah.
    Dia sendiri yang tentukan ada BOS atau tidak, FVG mana yang valid, range SH/SL.
    """
    h1_table = _candle_table(raw_data["h1"], limit=100)
    price = raw_data["price"]

    # Replay sudah dihandle di bot_core via ReplayEngine
    # ai_analysts hanya terima replay_text dari ctx kalau ada
    replay_text = ctx.get("replay_text", "") if ctx else ""

    # Baca rules dari JSON untuk inject ke prompt
    # Baca langsung dari file JSON — sumber kebenaran
    json_data = _load_json_files()
    ctx = json_data
    prompt_extra = json_data.get("prompts", {}).get("hiura", {}).get("extra_instructions", "")
    if prompt_extra and not prompt_ctx:
        prompt_ctx = prompt_extra
    logic = json_data.get("logic_raw", {})
    bos_cfg = logic.get("find_bos_h1", {})
    fvg_cfg = logic.get("find_fvg_h1", {})
    sw_left  = bos_cfg.get("swing_left",  8)
    sw_right = bos_cfg.get("swing_right", 8)
    sw_def   = bos_cfg.get("swing_definition", "")
    sw_ex    = bos_cfg.get("example", "")
    fvg_min  = fvg_cfg.get("min_gap_pct", 0.05)

    prompt = f"""Kamu adalah Hiura, analis struktur H1 ICT.
Harga sekarang: {price}

HASIL REPLAY H1 (Python sudah baca kiri ke kanan — ini yang paling akurat):
{replay_text}

DATA CANDLE H1 LENGKAP (untuk verifikasi):
{h1_table}

{_build_json_ctx(ctx)}

{('INSTRUKSI KATYUSHA: ' + prompt_ctx + chr(10)) if prompt_ctx else ''}TUGASMU — ikuti RULES dari JSON di atas:

1. BACA HASIL REPLAY di atas — Python sudah hitung swing valid dan BOS menggunakan rules swing_left={sw_left} swing_right={sw_right}.
   GUNAKAN hasil replay sebagai acuan utama. Verifikasi dengan candle kalau perlu.

2. CARI SWING LOW/HIGH VALID (konfirmasi dari replay):
   - swing_left={sw_left}: minimal {sw_left} candle SEBELUM candle X tidak ada yang lebih rendah (untuk swing low)
   - swing_right={sw_right}: minimal {sw_right} candle SESUDAH candle X tidak ada yang lebih rendah
   - Kedua syarat HARUS terpenuhi untuk swing valid
   {f"- {sw_def}" if sw_def else ""}
   {f"- CONTOH: {sw_ex}" if sw_ex else ""}

3. BOS H1 — gunakan level PERSIS dari replay di atas (swing_level field):
   - BOS bearish: cari swing low valid → ada candle yang CLOSE di bawah swing low itu
   - BOS bullish: cari swing high valid → ada candle yang CLOSE di atas swing high itu
   - Gunakan low/high PERSIS dari candle swing (field L: atau H: di tabel), JANGAN bulatkan
   - BOS terbentuk di LEVEL SWING yang ditembus, bukan di level candle yang break

4. FVG H1 — gunakan FVG dari replay di atas (sudah difilter arah dan min_gap):
   - FVG bearish: candle[i].L > candle[i+2].H — gap turun, harga tidak menyentuh zona ini
   - FVG bullish: candle[i].H < candle[i+2].L — gap naik
   - FVG harus SETELAH candle BOS terbentuk (idx > bos_candle_idx)
   - FVG filled = ada candle yang close menembus zona (wick masuk = belum filled)
   - Gap minimum {fvg_min}% dari harga

4. SWING RANGE setelah BOS:
   - BOS bearish: SH = high tertinggi SETELAH BOS candle, SL = swing low yang jadi BOS itu sendiri
   - BOS bullish: SL = low terendah SETELAH BOS candle, SH = swing high yang jadi BOS

5. WATCHLIST:
   - BOS bearish: level FVG bearish untuk retrace naik (harga naik sentuh FVG dari bawah)
   - BOS bullish: level FVG bullish untuk retrace turun
   - Gunakan harga PERSIS dari candle

Balas JSON murni:
{{
  "bias": "bullish|bearish|neutral",
  "bos_found": true,
  "bos_type": "bullish_bos|bearish_bos",
  "bos_level": 0.0,
  "bos_candle_idx": 0,
  "sh_since_bos": 0.0,
  "sl_before_bos": 0.0,
  "fvg_list": [
    {{"type": "bullish|bearish", "high": 0.0, "low": 0.0, "candle_idx": 0, "fresh": true}}
  ],
  "watchlist": [
    {{"level": 0.0, "condition": "touch", "reason": "FVG high/low @ X.XX"}}
  ],
  "chat_msg": "pesan ke grup WA max 2 kalimat",
  "confidence": 0.0
}}"""

    raw = _call(client, model, prompt, max_tokens=800, temp=0.2)
    parsed = _parse_json(raw)
    if not parsed:
        logger.warning(f"[HIURA] Parse gagal: {raw[:100]}")
        return {"bias": "neutral", "bos_found": False, "watchlist": [],
                "fvg_list": [], "chat_msg": "", "confidence": 0}

    logger.info(f"[HIURA] bias={parsed.get('bias')} bos={parsed.get('bos_found')} "
                f"bos_level={parsed.get('bos_level',0):.2f} fvg={len(parsed.get('fvg_list',[]))}")
    return parsed


# ══════════════════════════════════════════════════════════
# SENANAN (AI-2) — IDM Hunter M5
# Input: candle M5 mentah + range SH/SL dari Hiura
# Output: IDM ditemukan? Level watchlist?
# ══════════════════════════════════════════════════════════

def senanan_idm_hunt(client: Groq, model: str, raw_data: dict,
                     sh: float, sl: float, m5_idm_direction: str,
                     bias_h1: str, ctx: dict = None, prompt_ctx: str = "") -> dict:
    """
    Senanan cari IDM di M5 dalam range SH-SL.
    """
    # Baca langsung dari file JSON
    json_data = _load_json_files()
    ctx = json_data
    prompt_extra = json_data.get("prompts", {}).get("senanan", {}).get("extra_instructions", "")
    if prompt_extra and not prompt_ctx:
        prompt_ctx = prompt_extra
    logic = json_data.get("logic_raw", {})
    idm_cfg = logic.get("find_idm_m5", {})
    direction_cfg = idm_cfg.get(m5_idm_direction, {})
    gap_min = direction_cfg.get("gap_min_candles", 1)
    touch_rule = direction_cfg.get("touch_rule", "wick_or_body")
    m5_table = _candle_table(raw_data["m5"], limit=100)
    price = raw_data["price"]

    prompt = f"""Kamu adalah Senanan, spesialis IDM (Inducement) di M5 ICT.
Harga sekarang: {price}
Bias H1: {bias_h1}
Range yang diberikan Hiura: SH={sh} SL={sl}
Cari IDM arah: {m5_idm_direction} ({"bearish karena H1 bullish - M5 retrace turun" if m5_idm_direction == "bearish" else "bullish karena H1 bearish - M5 retrace naik"})

HASIL REPLAY M5 (Python sudah baca kiri ke kanan):
{replay_m5_text}

DATA CANDLE M5 (untuk verifikasi):
{m5_table}

{_build_json_ctx(ctx)}

{('INSTRUKSI KATYUSHA: ' + prompt_ctx + chr(10)) if prompt_ctx else ''}TUGASMU:
Cari IDM {m5_idm_direction} di M5 dalam range harga {sl}–{sh}.
PRIORITAS: Gunakan hasil replay Python di atas. Konfirmasi watch_level dan set watchlist.

DEFINISI IDM (verifikasi):
- IDM {m5_idm_direction}: 
  {"Candle A buat swing HIGH → minimal 1 candle tidak tembus high A → tembus → Watch level = LOW candle A" if m5_idm_direction == "bearish" else "Candle A buat swing LOW → gap → tembus → Watch level = HIGH candle A"}

- Cari dari candle terbaru (paling bawah tabel) ke atas
- IDM harus dalam range harga {sl}–{sh}
- Level watch (yang harus disentuh harga) = {"low candle A untuk IDM bearish" if m5_idm_direction == "bearish" else "high candle A untuk IDM bullish"}
- Sentuh = wick atau body menyentuh level watch

Jika tidak ada IDM dalam range, set idm_found=false.

Balas JSON murni:
{{
  "idm_found": true,
  "idm_type": "bullish_idm|bearish_idm",
  "candle_a_idx": 0,
  "candle_a_high": 0.0,
  "candle_a_low": 0.0,
  "watch_level": 0.0,
  "watch_condition": "touch",
  "watch_reason": "low/high candle A IDM @ X.XX",
  "watchlist": [
    {{"level": 0.0, "condition": "touch", "reason": "IDM watch level"}}
  ],
  "chat_msg": "pesan ke grup WA max 2 kalimat",
  "confidence": 0.0
}}"""

    raw = _call(client, model, prompt, max_tokens=700, temp=0.2)
    parsed = _parse_json(raw)
    if not parsed:
        logger.warning(f"[SENANAN] Parse gagal: {raw[:100]}")
        return {"idm_found": False, "watchlist": [], "chat_msg": "", "confidence": 0}

    logger.info(f"[SENANAN] idm={parsed.get('idm_found')} watch={parsed.get('watch_level',0):.2f}")
    return parsed


# ══════════════════════════════════════════════════════════
# SHINA (AI-3) — BOS/MSS Guard M5
# Input: candle M5 + konteks IDM yang disentuh
# Output: BOS? MSS? Lanjut ke entry atau reset?
# ══════════════════════════════════════════════════════════

def shina_bos_mss(client: Groq, model: str, raw_data: dict,
                  idm_info: dict, bias_h1: str,
                  sh: float, sl: float, prompt_ctx: str = "") -> dict:
    """
    Shina analisis BOS/MSS di M5 setelah IDM disentuh.
    """
    # Baca langsung dari file JSON
    json_data = _load_json_files()
    prompt_extra = json_data.get("prompts", {}).get("shina", {}).get("extra_instructions", "")
    if prompt_extra and not prompt_ctx:
        prompt_ctx = prompt_extra
    logic = json_data.get("logic_raw", {})
    bos_m5_cfg = logic.get("find_bos_m5", {})
    mss_m5_cfg = logic.get("find_mss_m5", {})
    m5_table = _candle_table(raw_data["m5"], limit=80)
    price = raw_data["price"]
    watch_level = idm_info.get("watch_level", 0)
    idm_type = idm_info.get("idm_type", "")

    prompt = f"""Kamu adalah Shina, penjaga BOS/MSS M5 ICT.
Harga sekarang: {price}
Bias H1: {bias_h1}
IDM M5 yang baru disentuh: {idm_type} @ watch level {watch_level}
Range aktif: SH={sh} SL={sl}

DATA CANDLE M5 (closed, terbaru di bawah):
{m5_table}

{('INSTRUKSI KATYUSHA: ' + prompt_ctx + chr(10)) if prompt_ctx else ''}TUGASMU:
Setelah IDM disentuh, cek apakah terjadi BOS atau MSS di M5.

DEFINISI:
- Freeze range = area dari candle IDM terbentuk sampai candle IDM disentuh
  Freeze high = highest high dalam range itu
  Freeze low  = lowest low dalam range itu

Untuk bias H1 {bias_h1}:
{"- BOS bearish M5: ada close CANDLE di bawah freeze low → lanjut ke entry sniper (tunggu MSS)" if bias_h1 == "bullish" else "- BOS bullish M5: ada close CANDLE di atas freeze high → lanjut ke entry sniper (tunggu MSS)"}
{"- MSS bullish M5: setelah BOS bearish, ada close di ATAS freeze high → entry signal!" if bias_h1 == "bullish" else "- MSS bearish M5: setelah BOS bullish, ada close di BAWAH freeze low → entry signal!"}
- Jika tidak ada BOS/MSS: keputusan = wait, set watchlist di freeze high/low

Tentukan freeze range dari candle terbaru dan beri keputusan.

Balas JSON murni:
{{
  "freeze_high": 0.0,
  "freeze_low": 0.0,
  "bos_found": false,
  "bos_type": "bearish_bos_m5|bullish_bos_m5",
  "mss_found": false,
  "mss_type": "bullish_mss_m5|bearish_mss_m5",
  "mss_candle_high": 0.0,
  "mss_candle_low": 0.0,
  "decision": "entry|wait|reset_idm",
  "watchlist": [
    {{"level": 0.0, "condition": "break_above|break_below|touch", "reason": "freeze high/low"}}
  ],
  "chat_msg": "pesan ke grup WA max 2 kalimat",
  "confidence": 0.0
}}"""

    raw = _call(client, model, prompt, max_tokens=700, temp=0.2)
    parsed = _parse_json(raw)
    if not parsed:
        logger.warning(f"[SHINA] Parse gagal: {raw[:100]}")
        return {"decision": "wait", "watchlist": [], "mss_found": False,
                "chat_msg": "", "confidence": 0}

    logger.info(f"[SHINA] decision={parsed.get('decision')} mss={parsed.get('mss_found')} "
                f"freeze={parsed.get('freeze_low',0):.2f}-{parsed.get('freeze_high',0):.2f}")
    return parsed


# ══════════════════════════════════════════════════════════
# YUSUF (AI-4) — Entry Sniper
# Input: semua konteks + FVG H1 + MSS candle
# Output: entry price, SL, TP, confidence
# ══════════════════════════════════════════════════════════

def yusuf_entry(client: Groq, model: str, raw_data: dict,
                hiura_data: dict, shina_data: dict,
                trade_memory: list, ctx: dict = None, prompt_ctx: str = "") -> dict:
    """
    Yusuf tentukan entry, SL, TP.
    Dia punya akses ke memory trade sebelumnya untuk belajar.
    """
    price = raw_data["price"]
    bias = hiura_data.get("bias", "neutral")
    fvg_list = hiura_data.get("fvg_list", [])
    mss_high = shina_data.get("mss_candle_high", 0)
    mss_low  = shina_data.get("mss_candle_low",  0)

    fvg_text = "\n".join([
        f"  FVG {f['type']}: {f['low']}–{f['high']}"
        for f in fvg_list if f.get("fresh", True)
    ]) or "  Tidak ada FVG fresh"

    mem_text = ""
    if trade_memory:
        mem_text = "TRADE SEBELUMNYA (belajar dari ini):\n"
        for t in trade_memory[-5:]:
            mem_text += (f"  [{t.get('result','?')}] {t.get('direction','?')} "
                        f"entry={t.get('entry',0)} setup={t.get('setup','?')} "
                        f"notes={t.get('notes','')[:50]}\n")

    # Baca langsung dari file JSON
    json_data = _load_json_files()
    ctx = json_data
    prompt_extra = json_data.get("prompts", {}).get("yusuf", {}).get("extra_instructions", "")
    if prompt_extra and not prompt_ctx:
        prompt_ctx = prompt_extra
    logic = json_data.get("logic_raw", {})
    sl_cfg = logic.get("stop_loss", {})
    tp_cfg = logic.get("take_profit", {})
    min_rr = tp_cfg.get("min_rr", 2.0)
    sl_buf = sl_cfg.get("buffer_pct", 0.0)

    prompt = f"""Kamu adalah Yusuf, entry sniper ICT.
Harga sekarang: {price}
Bias H1: {bias}
MSS candle: High={mss_high} Low={mss_low}

FVG H1 (hanya sebagai referensi zona, bukan syarat wajib):
{fvg_text}

{mem_text}
{_build_json_ctx(ctx)}

{('INSTRUKSI KATYUSHA: ' + prompt_ctx + chr(10)) if prompt_ctx else ''}TUGASMU:
Tentukan entry paling presisi. Gunakan angka PERSIS dari data, jangan bulatkan.

RULES ENTRY (dari logic_rules.json — WAJIB DIIKUTI):
- Entry saat MSS terkonfirmasi
- FVG H1 role: reference_only — info tambahan, bukan gate wajib
- Bias bullish: entry limit di area MSS candle close atau sedikit di bawahnya
- Bias bearish: entry limit di area MSS candle close atau sedikit di atasnya

RULES SL (dari logic_rules.json — WAJIB DIIKUTI):
- Bias bullish: SL = low candle MSS. Buffer: {{sl_buf}}%
- Bias bearish: SL = high candle MSS. Buffer: {{sl_buf}}%

RULES TP (dari logic_rules.json — WAJIB DIIKUTI):
- Bias bullish: TP = highest high sejak BOS H1
- Bias bearish: TP = lowest low sejak BOS H1
- Min RR = {{min_rr}} — skip kalau RR < {{min_rr}}

Balas JSON murni:
{{
  "decision": "entry|skip",
  "skip_reason": "",
  "direction": "buy|sell",
  "entry": 0.0,
  "sl": 0.0,
  "tp": 0.0,
  "rr": 0.0,
  "setup_type": "MSS_confirmed|MSS_in_FVG|MSS_outside_FVG|skip",
  "in_fvg_zone": true,
  "sl_reason": "low/high MSS candle @ X.XX",
  "tp_reason": "liquidity pool @ X.XX",
  "chat_msg": "pesan ke grup WA max 3 kalimat",
  "confidence": 0.0
}}"""

    raw = _call(client, model, prompt, max_tokens=500, temp=0.15)
    parsed = _parse_json(raw)
    if not parsed:
        logger.warning(f"[YUSUF] Parse gagal: {raw[:100]}")
        return {"decision": "skip", "chat_msg": "", "confidence": 0,
                "entry": 0, "sl": 0, "tp": 0}

    logger.info(f"[YUSUF] decision={parsed.get('decision')} "
                f"entry={parsed.get('entry',0):.2f} sl={parsed.get('sl',0):.2f} "
                f"tp={parsed.get('tp',0):.2f} rr={parsed.get('rr',0):.1f} "
                f"conf={parsed.get('confidence',0):.0%}")
    return parsed


# ══════════════════════════════════════════════════════════
# LOSS DEBRIEF — semua AI evaluasi setelah loss
# ══════════════════════════════════════════════════════════

def loss_debrief(clients: list, models: list,
                 closed_trade: dict, all_ai_data: dict) -> dict:
    """4 AI evaluasi trade yang loss dari perspektif masing-masing."""

    trade_str = (
        f"Direction: {closed_trade.get('direction','?')} "
        f"Entry: {closed_trade.get('entry',0):.2f} "
        f"SL: {closed_trade.get('sl',0):.2f} "
        f"Exit: {closed_trade.get('exit_price',0):.2f} "
        f"PnL: {closed_trade.get('pnl',0):.4f}"
    )

    ai_info = [
        ("Hiura",   "analisis H1 BOS dan FVG",       "Apakah BOS valid? FVG yang dipakai tepat?"),
        ("Senanan", "IDM M5",                          "Apakah IDM valid? Level watch sudah benar?"),
        ("Shina",   "BOS/MSS M5",                     "Apakah BOS/MSS keputusannya benar?"),
        ("Yusuf",   "entry SL TP",                    "Apakah entry, SL, TP sudah di level yang benar?"),
    ]

    chat_log = []
    verdicts = {}

    for i, (name, role, question) in enumerate(ai_info):
        if i >= len(clients):
            break
        prompt = f"""Kamu adalah {name}, spesialis {role}.
Trade yang loss: {trade_str}
Evaluasi: {question}
2-3 kalimat jujur. Tidak perlu defensif. Gaya WA.
Format: "{name}: [evaluasimu]" """

        raw = _call(clients[i], models[i], prompt, max_tokens=180, temp=0.5)
        verdicts[name.lower()] = raw
        chat_log.append({"ai": f"ai{i+1}", "nama": name, "pesan": raw, "ronde": 1})
        logger.info(f"[DEBRIEF {name}] {raw[:80]}")

    # Root cause analysis
    debrief_text = "\n".join([f"{k}: {v}" for k, v in verdicts.items()])
    rc_prompt = f"""Berdasarkan evaluasi 4 AI setelah loss:
{debrief_text}
Trade: {trade_str}

Identifikasi: siapa paling bertanggung jawab? root cause? rule baru apa yang harus diterapkan?
Balas JSON murni:
{{"culprit":"Hiura|Senanan|Shina|Yusuf","root_cause":"...","new_rule":"...","lesson":"..."}}"""

    rc_raw = _call(clients[0], models[0], rc_prompt, max_tokens=250, temp=0.2)
    rc = _parse_json(rc_raw) or {
        "culprit": "unknown", "root_cause": "parse error",
        "new_rule": "", "lesson": "review manual"
    }

    chat_log.append({
        "ai": "system", "nama": "Debrief",
        "pesan": f"Root cause: {rc.get('culprit')} — {rc.get('root_cause')} | Rule baru: {rc.get('new_rule')}",
        "ronde": 2
    })

    return {**verdicts, **rc, "chat_log": chat_log}


# ══════════════════════════════════════════════════════════
# KATYUSHA — Supervisor via OpenRouter (Claude Sonnet)
# Pantau setiap 5 jam + evaluasi setelah trade
# Authority: langsung override tanpa tanya ulang
# ══════════════════════════════════════════════════════════

def katyusha_review(openrouter_key: str, bot_state: dict,
                    raw_data: dict, all_ai_data: dict,
                    rules_current: dict = None,
                    logic_current: dict = None) -> dict:
    # Selalu baca langsung dari file — ini yang paling akurat
    json_data = _load_json_files()
    if not rules_current:
        rules_current = json_data.get("rules", {})
    if not logic_current:
        logic_current = json_data.get("logic", {})
    prompts_current = json_data.get("prompts", {})
    """
    Katyusha review market + analisis AI + rules + logic.
    Authority penuh: bisa edit/tambah/hapus rules dan logic langsung.

    Return: {
      "verdict": "ok|override|warning",
      "override_action": "none|reset|skip_entry|force_phase",
      "override_phase": "",
      "rules_changes": [...],
      "rules_adds": [...],
      "rules_removes": [...],
      "logic_changes": [...],
      "logic_adds": [...],
      "logic_removes": [...],
      "chat_msg": "...",
      "reasoning": "..."
    }
    """
    import requests

    phase   = bot_state.get("phase", "unknown")
    price   = raw_data.get("price", 0)
    h1_table = _candle_table(raw_data.get("h1", []), limit=50)
    m5_table = _candle_table(raw_data.get("m5", []), limit=40)

    hiura_sum   = json.dumps(all_ai_data.get("hiura",   {}), ensure_ascii=False)[:400]
    senanan_sum = json.dumps(all_ai_data.get("senanan", {}), ensure_ascii=False)[:300]
    shina_sum   = json.dumps(all_ai_data.get("shina",   {}), ensure_ascii=False)[:300]

    watchlist_text = "\n".join([
        f"  {w.get('condition','touch').upper()} @ {w.get('level',0):.2f} — {w.get('reason','')[:60]}"
        for w in bot_state.get("watchlist", [])
    ]) or "  (kosong)"

    # Kirim FULL JSON — Katyusha harus lihat semua, tidak boleh dipotong
    rules_text   = json.dumps(
        {k:v for k,v in (rules_current or {}).items() if not k.startswith("_")},
        ensure_ascii=False, indent=2
    )
    logic_text   = json.dumps(
        {k:v for k,v in (logic_current or {}).items() if not k.startswith("_")},
        ensure_ascii=False, indent=2
    )
    prompts_text = json.dumps(
        {k:v for k,v in prompts_current.items() if not k.startswith("_")},
        ensure_ascii=False, indent=2
    )

    prompt = f"""Kamu adalah Katyusha, supervisor ICT trading bot dengan authority penuh.
Kamu pakai Claude Sonnet — reasoning kamu lebih kuat dari AI Groq lain di tim.
Harga sekarang: {price} | Fase: {phase}

CANDLE H1 (closed, terbaru di bawah):
{h1_table}

CANDLE M5:
{m5_table}

ANALISIS AI SAAT INI:
Hiura (H1 analyst): {hiura_sum}
Senanan (IDM hunter): {senanan_sum}
Shina (BOS/MSS guard): {shina_sum}

WATCHLIST AKTIF:
{watchlist_text}

RULES SAAT INI (data/rules.json):
{rules_text}

LOGIC SAAT INI (data/logic_rules.json):
{logic_text}

PROMPTS SAAT INI (data/prompts.json — instruksi AI):
{prompts_text}

TUGASMU — cek semua secara menyeluruh:

1. VALIDASI ANALISIS AI:
   - Apakah BOS yang diklaim Hiura benar ada di candle? Cek level vs data candle
   - Apakah IDM Senanan masuk akal? Level IDM ada di data candle?
   - Apakah decision Shina konsisten dengan data?
   - Apakah watchlist level realistis vs harga sekarang?

2. VALIDASI RULES & LOGIC:
   - Apakah ada rules yang terlalu ketat/longgar untuk kondisi market saat ini?
   - Apakah logic (cara deteksi BOS/FVG/IDM) perlu penyesuaian?
   - Apakah ada rules/logic yang missing dan seharusnya ada?

3. TINDAKAN:
   - Kalau analisis AI salah → override (reset/force_phase/skip)
   - Kalau rules/logic perlu perbaikan → edit langsung sekarang
   - Bisa tambah rules baru, ubah yang ada, atau hapus yang tidak perlu
   - Kalau semua OK → verdict = "ok"

FORMAT PERUBAHAN:
- rules_changes: ubah nilai yang ada. Format: [{{"section":"entry","key":"min_confidence","new":0.7,"reason":"..."}}]
- rules_adds: tambah key baru. Format: [{{"section":"entry","key":"require_fvg","value":true,"reason":"..."}}]
- rules_removes: hapus key. Format: [{{"section":"filter","key":"skip_if_ranging","reason":"..."}}]
- Sama untuk logic_changes, logic_adds, logic_removes

Balas JSON murni:
{{
  "verdict": "ok|override|warning",
  "override_action": "none|reset|skip_entry|force_phase",
  "override_phase": "",
  "rules_changes": [],
  "rules_adds": [],
  "rules_removes": [],
  "logic_changes": [],
  "logic_adds": [],
  "logic_removes": [],
  "prompt_updates": [
    {{"ai": "hiura|senanan|shina|yusuf", "field": "extra_instructions|focus|style", "value": "..."}}
  ],
  "market_assessment": "kondisi market sekarang (1-2 kalimat)",
  "chat_msg": "pesan ke grup WA max 3 kalimat — tegas dan informatif",
  "reasoning": "penjelasan keputusan (max 120 kata)"
}}"""

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://bot-botan-ict.railway.app",
                "X-Title": "ICT Trading Bot",
            },
            json={
                "model": "anthropic/claude-sonnet-4-6",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.15,
                "max_tokens": 2000,
            },
            timeout=45,
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        parsed = _parse_json(raw)
        if not parsed:
            logger.warning(f"[KATYUSHA] Parse gagal: {raw[:100]}")
            return {"verdict": "ok", "override_action": "none", "chat_msg": "",
                    "rules_changes": [], "rules_adds": [], "rules_removes": [],
                    "logic_changes": [], "logic_adds": [], "logic_removes": []}

        total_changes = (len(parsed.get("rules_changes",[])) +
                         len(parsed.get("rules_adds",[])) +
                         len(parsed.get("rules_removes",[])) +
                         len(parsed.get("logic_changes",[])) +
                         len(parsed.get("logic_adds",[])) +
                         len(parsed.get("logic_removes",[])))
        logger.info(f"[KATYUSHA] verdict={parsed.get('verdict')} "
                    f"action={parsed.get('override_action')} "
                    f"changes={total_changes}")
        return parsed

    except Exception as e:
        logger.warning(f"[KATYUSHA] Error: {str(e)[:80]}")
        return {"verdict": "ok", "override_action": "none", "chat_msg": "",
                "rules_changes": [], "rules_adds": [], "rules_removes": [],
                "logic_changes": [], "logic_adds": [], "logic_removes": []}


def katyusha_post_trade(openrouter_key: str, closed_trade: dict,
                        all_ai_data: dict, debrief: dict,
                        rules_current: dict, logic_current: dict) -> dict:
    """
    Katyusha evaluasi mendalam setelah trade selesai.
    Output: apakah rules/logic perlu diubah + rekomendasi spesifik.
    """
    import requests

    prompt = f"""Kamu adalah Katyusha, supervisor trading ICT.
Trade yang baru selesai:
{json.dumps(closed_trade, ensure_ascii=False)[:300]}

Debrief dari tim AI:
Culprit: {debrief.get('culprit','')}
Root cause: {debrief.get('root_cause','')}
New rule suggested: {debrief.get('new_rule','')}

Rules saat ini (ringkasan):
{json.dumps({k:v for k,v in rules_current.items() if not k.startswith('_')}, ensure_ascii=False, indent=2)}

Logic saat ini (FULL):
{json.dumps({k:v for k,v in logic_current.items() if not k.startswith('_')}, ensure_ascii=False, indent=2)}

TUGASMU:
Berikan evaluasi mendalam:
1. Apakah debrief tim sudah tepat menunjuk culprit?
2. Rules/logic mana yang HARUS diubah?
3. Berikan rekomendasi perubahan yang KONKRET (section + key + value baru)

Balas JSON murni:
{{
  "agree_with_debrief": true,
  "real_culprit": "Hiura|Senanan|Shina|Yusuf|sistem",
  "rules_changes": [
    {{"section": "entry", "key": "min_confidence", "old": 0.6, "new": 0.7, "reason": "..."}}
  ],
  "logic_changes": [
    {{"section": "find_idm_m5", "key": "gap_min_candles", "old": 1, "new": 2, "reason": "..."}}
  ],
  "chat_msg": "pesan ke grup evaluasi (2-3 kalimat tegas)",
  "summary": "ringkasan evaluasi Katyusha"
}}"""

    try:
        resp = requests.post(
            "https://openrouter.ai/api/v1/chat/completions",
            headers={
                "Authorization": f"Bearer {openrouter_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://bot-botan-ict.railway.app",
            },
            json={
                "model": "anthropic/claude-sonnet-4-6",
                "messages": [{"role": "user", "content": prompt}],
                "temperature": 0.15,
                "max_tokens": 1500,
            },
            timeout=40,
        )
        raw = resp.json()["choices"][0]["message"]["content"].strip()
        parsed = _parse_json(raw)
        if not parsed:
            return {"agree_with_debrief": True, "rules_changes": [], "logic_changes": [], "chat_msg": ""}
        logger.info(f"[KATYUSHA POST] culprit={parsed.get('real_culprit')} changes={len(parsed.get('rules_changes',[]))+len(parsed.get('logic_changes',[]))}")
        return parsed
    except Exception as e:
        logger.warning(f"[KATYUSHA POST] Error: {str(e)[:80]}")
        return {"agree_with_debrief": True, "rules_changes": [], "logic_changes": [], "chat_msg": ""}
