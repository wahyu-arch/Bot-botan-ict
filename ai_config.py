"""
AIConfigLoader — Load konfigurasi AI dari data/ai/<name>.json.
Setiap AI punya file sendiri: prompt_template, output_schema, rules, extra_instructions.
Python tidak lagi hardcode prompt — semuanya dari JSON.
"""
import os, json, logging
from typing import Optional

logger = logging.getLogger(__name__)

AI_CONFIG_DIR = "data/ai"
_cache: dict = {}  # cache per AI name


def load(ai_name: str, reload: bool = False) -> dict:
    """
    Load config AI dari data/ai/<ai_name>.json.
    Di-cache setelah pertama kali load.
    Set reload=True untuk force reload (setelah Katyusha edit file).
    """
    if ai_name in _cache and not reload:
        return _cache[ai_name]

    path = os.path.join(AI_CONFIG_DIR, f"{ai_name}.json")
    if not os.path.exists(path):
        # Auto-create dari bundled defaults di dalam package
        _create_default(ai_name)
        if not os.path.exists(path):
            logger.warning(f"[AI_CONFIG] File tidak ada dan gagal dibuat: {path}")
            return {}
    try:
        with open(path, encoding="utf-8") as f:
            cfg = json.load(f)
        _cache[ai_name] = cfg
        logger.debug(f"[AI_CONFIG] Loaded {ai_name} v{cfg.get('_version',1)}")
        return cfg
    except Exception as e:
        logger.error(f"[AI_CONFIG] Gagal load {path}: {e}")
        return {}



def _create_default(ai_name: str):
    """
    Buat file config default kalau belum ada di volume.
    Ini penting saat pertama deploy ke Railway dengan volume baru.
    """
    os.makedirs(AI_CONFIG_DIR, exist_ok=True)
    path = os.path.join(AI_CONFIG_DIR, f"{ai_name}.json")

    # Coba copy dari data_defaults/ai/ (bundled di repo)
    for src_dir in ["data_defaults/ai", "data/ai_defaults"]:
        src = os.path.join(src_dir, f"{ai_name}.json")
        if os.path.exists(src):
            import shutil
            shutil.copy(src, path)
            logger.info(f"[AI_CONFIG] Created {ai_name}.json from {src}")
            return

    # Buat minimal default kalau tidak ada source
    defaults = {
        "hiura":   {"prompt_template": "Kamu Hiura, analis H1 ICT.\n\n{state_ctx}\n\n{candle_table}\n\n{extra_instructions}\n\nBALAS HANYA JSON:", "rules": {"candle_limit_h1": 80}},
        "senanan": {"prompt_template": "Kamu Senanan, IDM hunter M5.\n\n{state_ctx}\n\n{candle_table}\n\n{extra_instructions}\n\nBALAS HANYA JSON:", "rules": {"candle_limit_m5": 60}},
        "shina":   {"prompt_template": "Kamu Shina, penjaga MSS M5.\n\n{state_ctx}\n\n{candle_table}\n\n{extra_instructions}\n\nBALAS HANYA JSON:", "rules": {"candle_limit_m5": 60}},
        "yusuf":   {"prompt_template": "Kamu Yusuf, entry sniper.\n\nHarga: {price}\n\n{state_ctx}\n\n{candle_table}\n\n{extra_instructions}\n\nBALAS HANYA JSON:", "rules": {"candle_limit_m5": 20}},
        "katyusha":{"prompt_template": "Kamu Katyusha, supervisor.\n\n{state_ctx}\n\n{extra_instructions}\n\nBALAS HANYA JSON:", "rules": {}},
    }
    cfg = defaults.get(ai_name, {})
    cfg["_version"] = 1
    cfg["_ai"] = ai_name
    cfg.setdefault("extra_instructions", "")
    cfg.setdefault("output_schema", {})
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        logger.info(f"[AI_CONFIG] Created minimal default: {ai_name}.json")
    except Exception as e:
        logger.error(f"[AI_CONFIG] Gagal buat default {ai_name}: {e}")


def build_prompt(ai_name: str, variables: dict) -> str:
    """
    Build prompt dari template di JSON config.
    variables: dict key → value untuk di-replace di {placeholder}.

    Placeholder yang tersedia:
      {state_ctx}         — state trading saat ini
      {candle_table}      — tabel candle H1 atau M5
      {extra_instructions}— instruksi tambahan dari JSON
      {notify_ctx}        — notifikasi dari AI lain
      {price}             — harga sekarang
      {bias}              — bias H1
      {watchlist_summary} — ringkasan watchlist aktif
      {schema}            — output schema (auto-inject)
    """
    cfg = load(ai_name)
    if not cfg:
        return ""

    template = cfg.get("prompt_template", "")
    extra    = cfg.get("extra_instructions", "")
    schema   = cfg.get("output_schema", {})

    # Inject schema sebagai contoh output wajib
    schema_str = json.dumps(schema, indent=2, ensure_ascii=False)

    # Build final variables
    ctx = {
        "extra_instructions": extra,
        "schema": f"FORMAT OUTPUT WAJIB (balas HANYA JSON ini):\n{schema_str}",
        "notify_ctx": "",
        "watchlist_summary": "",
        "price": "",
        "bias": "",
        "state_ctx": "",
        "candle_table": "",
    }
    ctx.update(variables)  # override dengan nilai aktual

    # Replace semua placeholder
    prompt = template
    for key, val in ctx.items():
        prompt = prompt.replace(f"{{{key}}}", str(val) if val else "")

    # Append schema di akhir
    if "{schema}" not in template:
        prompt += f"\n\n{ctx['schema']}"

    return prompt


def get_rules(ai_name: str) -> dict:
    """Ambil bagian rules dari config AI."""
    return load(ai_name).get("rules", {})


def get_candle_limit(ai_name: str, timeframe: str = "h1") -> int:
    """Ambil limit candle untuk AI dan timeframe tertentu."""
    rules = get_rules(ai_name)
    key   = f"candle_limit_{timeframe}"
    return rules.get(key, 60)


def invalidate_cache(ai_name: Optional[str] = None):
    """Hapus cache — dipanggil setelah Katyusha update file JSON."""
    global _cache
    if ai_name:
        _cache.pop(ai_name, None)
    else:
        _cache.clear()
    logger.info(f"[AI_CONFIG] Cache invalidated: {ai_name or 'all'}")


def save(ai_name: str, updates: dict) -> bool:
    """
    Update field tertentu di data/ai/<ai_name>.json.
    Dipanggil oleh Katyusha saat ingin ubah extra_instructions atau rules AI lain.
    """
    os.makedirs(AI_CONFIG_DIR, exist_ok=True)
    path = os.path.join(AI_CONFIG_DIR, f"{ai_name}.json")
    try:
        cfg = load(ai_name) or {}
        for key, val in updates.items():
            if "." in key:
                # Nested key, e.g. "rules.min_rr"
                parts = key.split(".", 1)
                cfg.setdefault(parts[0], {})[parts[1]] = val
            else:
                cfg[key] = val
        cfg["_version"] = cfg.get("_version", 1) + 1
        with open(path, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2, ensure_ascii=False)
        invalidate_cache(ai_name)
        logger.info(f"[AI_CONFIG] {ai_name}.json saved v{cfg['_version']}")
        return True
    except Exception as e:
        logger.error(f"[AI_CONFIG] Gagal save {ai_name}: {e}")
        return False
