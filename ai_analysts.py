"""
AI Analysts — 4 AI tim trading. Dioptimalkan untuk Qwen di Groq.
JSON parsing robust, temperature rendah, prompt ringkas.
"""
import json
import logging
import re
from groq import Groq

logger = logging.getLogger(__name__)

JSON_INSTRUCTION = (
    "\n⚠️ OUTPUT WAJIB: JSON MURNI. TANPA markdown, TANPA teks penjelasan. "
    "Gunakan double quote untuk semua key/value. JANGAN trailing comma. "
    "Jika ragu, set confidence=0.0 dan decision='skip'."
)

def _sanitize_json(raw: str) -> str:
    if not raw: return ""
    raw = re.sub(r'```(?:json)?\s*', '', raw)
    raw = re.sub(r'```\s*', '', raw)
    raw = re.sub(r',\s*([}\]])', r'\1', raw)
    raw = re.sub(r'([{,]\s*)(\w+)(\s*:)', r'\1"\2"\3', raw)
    return raw.strip()

def _parse_json(raw: str) -> dict | list | None:
    cleaned = _sanitize_json(raw)
    match = re.search(r'(\{[\s\S]*\}|\[[\s\S]*\])', cleaned)
    if not match: return None
    try: return json.loads(match.group(0))
    except json.JSONDecodeError: return None

def _call(client: Groq, model: str, prompt: str, max_tokens: int = 600, temp: float = 0.1) -> str:
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=max_tokens,
            response_format={"type": "json_object"}
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        logger.warning(f"[API] {model}: {str(e)[:100]}")
        return ""

def _call_with_retry(client, model: str, prompt: str, max_tokens: int = 600, temp: float = 0.1, retries: int = 2) -> dict | None:
    for i in range(retries + 1):
        raw = _call(client, model, prompt + (JSON_INSTRUCTION if i < retries else ""), max_tokens, temp)
        parsed = _parse_json(raw)
        if parsed and isinstance(parsed, dict):
            parsed["_raw"] = raw
            return parsed
        if i < retries:
            logger.warning(f"[RETRY {i+1}] JSON invalid, forcing strict mode...")
    return None

def _candle_table(candles: list, limit: int = 30) -> str:
    return "\n".join([f"[{i}] H:{c['h']} L:{c['l']} C:{c['c']}" for i, c in enumerate(candles[-limit:])])

# ─────────────────────────────────────────────────────────────
# HIURA (AI-1)
# ─────────────────────────────────────────────────────────────
def hiura_h1_analysis(client, model, raw_data, ctx, handoff=None):
    h1_tbl = _candle_table(raw_data["h1"])
    prompt = f"""Kamu Hiura (Analis H1 ICT). Harga: {raw_data['price']}
DATA: {h1_tbl}
{ctx}
{f'KONTEKS SEBELUMNYA: {handoff}' if handoff else ''}
TUGAS: Identifikasi BOS H1 + FVG valid + range SH/SL.
Balas JSON:
{{"bias":"bullish|bearish|neutral","bos_found":true,"bos_level":0.0,"sh":0.0,"sl":0.0,"fvg_list":[{"type":"","high":0.0,"low":0.0,"filled":false}],"watchlist":[{"level":0.0,"condition":"touch","reason":""}],"chat_msg":"","confidence":0.0}}{JSON_INSTRUCTION}"""
    return _call_with_retry(client, model, prompt, 700, 0.12)

# ─────────────────────────────────────────────────────────────
# SENANAN (AI-2)
# ─────────────────────────────────────────────────────────────
def senanan_idm_hunt(client, model, raw_data, sh, sl, m5_dir, bias_h1, ctx, handoff=None):
    m5_tbl = _candle_table(raw_data["m5"])
    prompt = f"""Kamu Senanan (IDM Hunter M5). Range SH={sh} SL={sl} | Bias={bias_h1}
Cari IDM {m5_dir} dalam range.
DATA: {m5_tbl}
{ctx}
{f'KONTEKS: {handoff}' if handoff else ''}
Balas JSON:
{{"idm_found":true,"watch_level":0.0,"watchlist":[{"level":0.0,"condition":"touch","assigned_to":"shina"}],"actions":[{"type":"force_phase","phase":"bos_guard"}],"chat_msg":"","confidence":0.0}}{JSON_INSTRUCTION}"""
    return _call_with_retry(client, model, prompt, 600, 0.12)

# ─────────────────────────────────────────────────────────────
# SHINA (AI-3)
# ─────────────────────────────────────────────────────────────
def shina_bos_mss(client, model, raw_data, idm_info, bias_h1, sh, sl, ctx, handoff=None):
    m5_tbl = _candle_table(raw_data["m5"])
    prompt = f"""Kamu Shina (Guard MSS/BOS M5). IDM disentuh @{idm_info.get('watch_level')}
DATA: {m5_tbl} | Freeze range: {sh}-{sl}
{ctx}
{f'KONTEKS: {handoff}' if handoff else ''}
Cek BOS/MSS. Jika valid → entry. Jika gagal → reset.
Balas JSON:
{{"decision":"entry|wait|reset_idm","freeze_high":0.0,"freeze_low":0.0,"watchlist":[{"level":0.0,"condition":"","assigned_to":"yusuf"}],"actions":[{"type":"force_phase","phase":"entry_sniper"}],"chat_msg":"","confidence":0.0}}{JSON_INSTRUCTION}"""
    return _call_with_retry(client, model, prompt, 600, 0.12)

# ─────────────────────────────────────────────────────────────
# YUSUF (AI-4)
# ─────────────────────────────────────────────────────────────
def yusuf_entry(client, model, raw_data, hiura, shina, trade_mem, ctx, handoff=None):
    m5_tbl = _candle_table(raw_data["m5"], 20)
    prompt = f"""Kamu Yusuf (Entry Sniper). Bias={hiura.get('bias')} | MSS Candle: H={shina.get('mss_candle_high')} L={shina.get('mss_candle_low')}
DATA: {m5_tbl}
{ctx}
{f'KONTEKS: {handoff}' if handoff else ''}
Hitung entry presisi. RR minimal 1.5. Presisi 5 desimal.
Balas JSON:
{{"decision":"entry|skip","direction":"buy|sell","entry":0.0,"sl":0.0,"tp":0.0,"rr":0.0,"confidence":0.0,"chat_msg":"","skip_reason":""}}{JSON_INSTRUCTION}"""
    return _call_with_retry(client, model, prompt, 500, 0.08)