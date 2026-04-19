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
        return {}

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
