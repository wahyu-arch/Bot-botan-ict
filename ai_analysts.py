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

import ai_config as _ai_cfg


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
    """
    Context minimal untuk AI — hanya state trading aktif.
    Rules/logic/prompts sudah ada di data/ai/<name>.json, tidak perlu dikirim ulang.
    Target: < 300 token total untuk context ini.
    """
    if not ctx:
        return ""
    import json as _j
    parts = []

    # State trading saat ini
    if ctx.get("state"):
        parts.append(ctx["state"])

    # Warning kalau stuck
    rc  = ctx.get("reset_count", 0)
    cyc = ctx.get("cycles_in_phase", 0)
    if rc > 0 or cyc > 3:
        parts.append(f"⚠️ reset={rc} cycles={cyc}")

    # State dari AI lain — field penting saja, 1 baris per AI
    key_fields = ["bias","bos_level","bos_type","sh_since_bos","sl_before_bos",
                  "watch_level","idm_watch_level","freeze_range","mss_candle",
                  "decision","next_phase"]
    for data_key, label in [("hiura_data","H"),("senanan_data","S"),("shina_data","Sh")]:
        d_str = ctx.get(data_key, "{}")
        if not d_str or d_str == "{}":
            continue
        try:
            d = _j.loads(d_str)
            summary = {k: d[k] for k in key_fields if k in d and d[k] not in (None, 0, "", [])}
            if summary:
                parts.append(f"{label}:{_j.dumps(summary, separators=(',',':'))}")
        except Exception:
            pass

    return "\n".join(p for p in parts if p) if parts else ""


# ── Helpers ─────────────────────────────────────────────

def _strip_thinking(text: str) -> tuple[str, str]:
    """
    Pisahkan <think>...</think> block dari response Qwen QwQ.
    Return (thinking_content, clean_response)
    """
    import re as _re
    thinking = ""
    m = _re.search(r"<think>(.*?)</think>", text, _re.DOTALL)
    if m:
        thinking = m.group(1).strip()
        text = _re.sub(r"<think>.*?</think>", "", text, flags=_re.DOTALL).strip()
    return thinking, text


def _call(client: Groq, model: str, prompt: str,
          max_tokens: int = 500, temp: float = 0.3) -> str:
    """
    Panggil Groq API. Support Qwen QwQ thinking mode:
    - <think>...</think> block otomatis di-strip dari response
    - Raw response (dengan thinking) disimpan di _last_thinking global untuk debug
    """
    global _last_thinking_block
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=max_tokens,
        )
        raw = resp.choices[0].message.content.strip()
        thinking, clean = _strip_thinking(raw)
        if thinking:
            _last_thinking_block = thinking
            logger.debug(f"[THINK] {model}: {len(thinking)} chars thinking stripped")
        return clean
    except Exception as e:
        err = str(e)
        tag = "[RATE LIMIT]" if ("429" in err or "rate_limit" in err.lower() or "413" in err) else "[API ERROR]"
        logger.warning(f"{tag} {model}: {err[:100]}")
        return ""

_last_thinking_block: str = ""  # debug — isi thinking Qwen terakhir


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


def _is_thinking_model(model: str) -> bool:
    """Cek apakah model ini adalah thinking model (Qwen QwQ, dll)."""
    return any(x in model.lower() for x in ("qwq", "qwen", "deepseek-r", "r1"))


def _call_with_retry(client, model: str, prompt: str,
                     max_tokens: int = 800, temp: float = 0.2,
                     max_retries: int = 1) -> dict | None:
    """
    Panggil AI dan retry sampai dapat JSON valid.
    Untuk Qwen QwQ thinking model:
    - temperature harus 0.6 (Groq requirement untuk QwQ)
    - max_tokens dinaikkan otomatis untuk beri ruang thinking
    - <think> block sudah di-strip di _call()
    """
    # Qwen QwQ requirement: temp harus >= 0.5, recommended 0.6
    if _is_thinking_model(model):
        temp = max(temp, 0.6)
        # Beri ruang extra untuk thinking (thinking token tidak masuk output quota
        # tapi API butuh space — minimal 1024 extra)
        max_tokens = max(max_tokens, 1024)

    # Pastikan prompt selalu diakhiri dengan instruksi JSON yang tegas
    json_instruction = (
        "\n\n/no_think\n⚠️ OUTPUT HARUS JSON MURNI. "
        "Tidak ada teks, tidak ada ```json, tidak ada penjelasan. "
        "Langsung mulai dengan { dan akhiri dengan }."
        if _is_thinking_model(model) else
        "\n\n⚠️ Balas HANYA dengan JSON murni. Mulai langsung dengan { tanpa teks apapun."
    )
    if json_instruction.strip() not in prompt:
        prompt = prompt + json_instruction

    for attempt in range(max_retries + 1):
        try:
            raw = _call(client, model, prompt, max_tokens=max_tokens, temp=temp)
            if not raw:
                if attempt < max_retries:
                    logger.warning(f"[RETRY {attempt+1}/{max_retries}] Response kosong")
                continue
            parsed = _parse_json(raw)
            if parsed and isinstance(parsed, dict):
                parsed["_raw"] = raw
                return parsed
            if attempt < max_retries:
                logger.warning(f"[RETRY {attempt+1}/{max_retries}] JSON tidak valid: {raw[:80]}")
                # Retry dengan instruksi lebih keras
                prompt = (
                    "Output HANYA JSON ini (isi nilai yang sesuai):\n"
                    + raw[:200]  # tunjukkan sebagian response sebelumnya sebagai contoh
                    + "\n\nBALAS DENGAN JSON LENGKAP DAN VALID:"
                )
        except Exception as e:
            logger.warning(f"[RETRY {attempt+1}] Error: {e}")
    logger.error("[RETRY] Semua percobaan gagal — return None")
    return None


def _build_notify_ctx(ctx: dict) -> str:
    """Format notifikasi dari AI lain untuk dimasukkan ke prompt."""
    import json as _j
    parts = []
    for key, label in [("hiura_data","Hiura"),("senanan_data","Senanan"),("shina_data","Shina")]:
        d = ctx.get(key, "{}")
        if d and d != "{}":
            try:
                parsed = _j.loads(d)
                msg = parsed.get("chat_msg") or parsed.get("analysis","")
                if msg:
                    parts.append(f"[Dari {label}]: {msg[:120]}")
            except Exception:
                pass
    return "\n".join(parts) if parts else ""


def _candle_table(candles: list, limit: int = 40) -> str:
    """
    Format candle OHLC ultra-ringkas: idx|O|H|L|C
    Contoh baris: 173|0.20543|0.20671|0.20412|0.20634
    AI baca kiri ke kanan — tidak perlu replay engine.
    ~60% lebih hemat token vs format sebelumnya.
    """
    rows = ["idx|O|H|L|C"]
    for c in candles[-limit:]:
        rows.append(f"{c['i']}|{c['o']}|{c['h']}|{c['l']}|{c['c']}")
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
    h1_table = _candle_table(raw_data["h1"], limit=80)
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

    # Prompt dari data/ai/hiura.json — tidak hardcode di Python
    candle_limit = min(_ai_cfg.get_candle_limit("hiura", "h1"), 50)
    h1_table = _candle_table(raw_data["h1"], limit=candle_limit)
    notify_ctx = _build_notify_ctx(ctx)
    prompt = _ai_cfg.build_prompt("hiura", {
        "state_ctx":    _build_json_ctx(ctx),
        "candle_table": f"CANDLE H1 ({candle_limit} terakhir):\n{h1_table}",
        "notify_ctx":   notify_ctx,
        "price":        str(raw_data.get("price", "")),
    })
    if not prompt:
        logger.warning("[HIURA] Gagal build prompt dari hiura.json — fallback")
        prompt = f"Kamu Hiura. Analisis BOS H1. Harga: {raw_data.get('price')}. Balas JSON."


    parsed = _call_with_retry(client, model, prompt, max_tokens=800, temp=0.2)
    if not parsed:
        logger.warning("[HIURA] Parse gagal setelah retry")
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
    m5_table = _candle_table(raw_data["m5"], limit=60)
    price = raw_data["price"]

    # Jalankan replay M5 Python dulu — hasilnya dikirim ke AI sebagai konteks
    try:
        from candle_replay import ReplayEngine, format_replay_for_ai
        _tmp_engine = ReplayEngine()
        m5_replay_event = _tmp_engine.replay_m5(
            raw_data["m5"], m5_idm_direction, logic, sh, sl
        )
        replay_m5_text = format_replay_for_ai(m5_replay_event, _tmp_engine)
    except Exception as _e:
        logger.warning(f"[SENANAN] replay_m5 error: {_e}")
        replay_m5_text = f"(Replay M5 tidak tersedia: {_e})"

    # Prompt dari data/ai/senanan.json
    candle_limit = min(_ai_cfg.get_candle_limit("senanan", "m5"), 40)
    m5_table = _candle_table(raw_data["m5"], limit=candle_limit)
    notify_ctx = _build_notify_ctx(ctx)
    prompt = _ai_cfg.build_prompt("senanan", {
        "state_ctx":    _build_json_ctx(ctx),
        "candle_table": f"CANDLE M5 ({candle_limit} terakhir):\n{m5_table}",
        "notify_ctx":   notify_ctx,
    })
    if not prompt:
        prompt = f"Kamu Senanan. Cari IDM M5 di range SH-SL. Harga: {raw_data.get('price')}. Balas JSON."


    parsed = _call_with_retry(client, model, prompt, max_tokens=700, temp=0.2)
    if not parsed:
        logger.warning("[SENANAN] Parse gagal setelah retry")
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
    m5_table = _candle_table(raw_data["m5"], limit=60)
    price = raw_data["price"]
    watch_level = idm_info.get("watch_level", 0)
    idm_type = idm_info.get("idm_type", "")

    # Prompt dari data/ai/shina.json
    candle_limit = min(_ai_cfg.get_candle_limit("shina", "m5"), 40)
    m5_table = _candle_table(raw_data["m5"], limit=candle_limit)
    notify_ctx = _build_notify_ctx(ctx)
    prompt = _ai_cfg.build_prompt("shina", {
        "state_ctx":    _build_json_ctx(ctx),
        "candle_table": f"CANDLE M5 ({candle_limit} terakhir):\n{m5_table}",
        "notify_ctx":   notify_ctx,
    })
    if not prompt:
        prompt = f"Kamu Shina. Cek MSS M5 freeze range. Harga: {raw_data.get('price')}. Balas JSON."


    parsed = _call_with_retry(client, model, prompt, max_tokens=700, temp=0.2)
    if not parsed:
        logger.warning("[SHINA] Parse gagal setelah retry")
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

    # Ambil data M5 terbaru untuk Yusuf bisa lihat level persis
    m5_recent = _candle_table(raw_data.get("m5", []), limit=20)
    sh_since_bos = hiura_data.get("sh_since_bos", 0)
    sl_before_bos = hiura_data.get("sl_before_bos", 0)
    freeze_high = shina_data.get("freeze_high", 0)
    freeze_low  = shina_data.get("freeze_low",  0)

    # Prompt dari data/ai/yusuf.json
    candle_limit = min(_ai_cfg.get_candle_limit("yusuf", "m5"), 15)
    m5_recent = _candle_table(raw_data.get("m5", []), limit=candle_limit)
    notify_ctx = _build_notify_ctx(ctx)
    prompt = _ai_cfg.build_prompt("yusuf", {
        "state_ctx":    _build_json_ctx(ctx),
        "candle_table": f"CANDLE M5 ({candle_limit} terakhir):\n{m5_recent}",
        "notify_ctx":   notify_ctx,
        "price":        str(raw_data.get("price", "")),
        "bias":         hiura_data.get("bias", ""),
    })
    if not prompt:
        prompt = f"Kamu Yusuf. Hitung entry/SL/TP dari MSS candle. Harga: {raw_data.get('price')}. Balas JSON."


    parsed = _call_with_retry(client, model, prompt, max_tokens=600, temp=0.15)
    if not parsed:
        logger.warning("[YUSUF] Parse gagal setelah retry")
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

FASE YANG TERSEDIA untuk force_phase:
- "h1_scan"     → Hiura scan ulang BOS H1
- "fvg_wait"    → tunggu FVG H1 disentuh
- "idm_hunt"    → Senanan cari IDM M5 (perlu bos & fvg di state)
- "bos_guard"   → Shina pantau BOS/MSS M5
- "entry_sniper" → Yusuf hitung entry sekarang

Balas JSON murni:
{{
  "verdict": "ok|override|warning",
  "override_action": "none|reset|skip_entry|force_phase|force_idm_hunt|force_entry",
  "override_phase": "h1_scan|fvg_wait|idm_hunt|bos_guard|entry_sniper",
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
