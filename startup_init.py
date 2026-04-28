"""
startup_init.py — Inisialisasi file JSON dari data_defaults/ ke data/ (volume).

Logika:
- data_defaults/ = dari GitHub (source of truth untuk fresh deploy)
- data/ = Railway Volume (diupdate AI saat runtime)
- Saat volume kosong/baru: copy dari data_defaults/ ke data/
- Saat file sudah ada di data/: TIDAK overwrite (AI updates dipertahankan)
"""
import os, shutil, json, logging

logger = logging.getLogger(__name__)

DEFAULTS_DIR = "data_defaults"
DATA_DIR     = "data"
FILES = ["rules.json", "logic_rules.json", "prompts.json"]


def init_data_files():
    """Copy file defaults ke data/ kalau belum ada. Tidak overwrite yang sudah ada."""
    os.makedirs(DATA_DIR, exist_ok=True)
    copied = []
    skipped = []

    for fname in FILES:
        src = os.path.join(DEFAULTS_DIR, fname)
        dst = os.path.join(DATA_DIR, fname)

        if os.path.exists(dst):
            # File sudah ada — cek apakah valid JSON
            try:
                with open(dst) as f:
                    json.load(f)
                skipped.append(fname)
                continue
            except Exception:
                logger.warning(f"[INIT] {fname} korup — replace dengan default")

        if os.path.exists(src):
            shutil.copy2(src, dst)
            copied.append(fname)
            logger.info(f"[INIT] Copied {fname} dari data_defaults/")
        else:
            logger.warning(f"[INIT] Default tidak ada: {src}")

    if copied:
        logger.info(f"[INIT] File diinisialisasi: {copied}")
    if skipped:
        logger.info(f"[INIT] File dipertahankan (sudah ada): {skipped}")

    # Init AI config files — data/ai/ akan di-create otomatis oleh ai_config.py
    # Tidak perlu copy di sini karena ai_config._create_default() handle itu
    os.makedirs("data/ai", exist_ok=True)

    return {"copied": copied, "skipped": skipped}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    result = init_data_files()
    print(result)
