"""
PromptEngine — Load dan serve prompt instruksi untuk setiap AI.
Katyusha bisa edit data/prompts.json untuk ubah perilaku AI secara real-time.
"""
import os, json, logging
from datetime import datetime, timezone

logger = logging.getLogger(__name__)
PROMPTS_FILE = "data/prompts.json"


class PromptEngine:
    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self._prompts = self._load()

    def _load(self) -> dict:
        if os.path.exists(PROMPTS_FILE):
            try:
                with open(PROMPTS_FILE) as f:
                    p = json.load(f)
                logger.info(f"[PROMPTS] Loaded v{p.get('_version',1)}")
                return p
            except Exception as e:
                logger.error(f"[PROMPTS] Load error: {e}")
        # File tidak ada — buat default dan simpan
        default = self._default()
        try:
            with open(PROMPTS_FILE, "w") as f:
                json.dump(default, f, indent=2, ensure_ascii=False)
            logger.info("[PROMPTS] Default prompts.json dibuat")
        except Exception as e:
            logger.warning(f"[PROMPTS] Gagal simpan default: {e}")
        return default

    def _default(self) -> dict:
        return {
            "_version": 1,
            "_last_updated": None,
            "_update_reason": "Auto-generated default",
            "hiura": {
                "focus": "Analisis struktur H1: BOS, FVG, bias market",
                "style": "Singkat, teknikal, pakai angka persis dari candle",
                "extra_instructions": ""
            },
            "senanan": {
                "focus": "Cari IDM M5: identifikasi swing, gap, level watch",
                "style": "State machine, langkah per langkah",
                "extra_instructions": ""
            },
            "shina": {
                "focus": "Konfirmasi BOS/MSS M5 setelah IDM disentuh",
                "style": "Tegas: entry atau wait, sertakan alasan",
                "extra_instructions": ""
            },
            "yusuf": {
                "focus": "Entry presisi: tentukan entry, SL, TP dengan angka persis",
                "style": "Entry sniper, percaya diri, RR minimal 2:1",
                "extra_instructions": ""
            },
            "katyusha": {
                "focus": "Supervisor: validasi semua AI, override kalau perlu",
                "style": "Otoritatif, langsung ke poin, bahasa Indonesia",
                "extra_instructions": ""
            }
        }

    def reload(self):
        self._prompts = self._load()

    def get(self, ai_name: str) -> dict:
        return self._prompts.get(ai_name.lower(), {})

    def build_context(self, ai_name: str) -> str:
        """Return string konteks untuk disertakan di prompt AI."""
        d = self.get(ai_name)
        if not d:
            return ""
        parts = []
        if d.get("focus"):
            parts.append(f"FOKUSMU: {d['focus']}")
        if d.get("style"):
            parts.append(f"GAYA KOMUNIKASI: {d['style']}")
        if d.get("extra_instructions"):
            parts.append(f"INSTRUKSI TAMBAHAN: {d['extra_instructions']}")
        return "\n".join(parts) if parts else ""

    def save(self, prompts: dict):
        with open(PROMPTS_FILE, "w") as f:
            json.dump(prompts, f, indent=2, ensure_ascii=False)
        self._prompts = prompts
        logger.info(f"[PROMPTS] Saved v{prompts.get('_version',1)} — {prompts.get('_update_reason','')}")

    @property
    def prompts(self) -> dict:
        return self._prompts
