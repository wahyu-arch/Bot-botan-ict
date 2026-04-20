"""
LogicEngine — Sistem rules yang bisa diedit AI.

Setiap fungsi trading (find_bos, find_fvg, find_idm, entry, sl, tp)
dibaca dari logic_rules.json. AI bisa update rules ini setelah loss
untuk mengubah CARA KERJA bot, bukan hanya parameter angkanya.

Berbeda dengan rules_engine.py (parameter angka),
logic_engine.py mengubah LOGIKA (kondisi, metode, cara deteksi).
"""

import os
import json
import logging
from datetime import datetime, timezone
from groq import Groq

logger = logging.getLogger(__name__)

LOGIC_FILE = "data/logic_rules.json"
LOGIC_HISTORY_FILE = "data/logic_rules_history.json"


class LogicEngine:
    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self._rules = self._load()

    def _load(self) -> dict:
        if os.path.exists(LOGIC_FILE):
            try:
                with open(LOGIC_FILE) as f:
                    rules = json.load(f)
                logger.info(f"[LOGIC] Loaded v{rules.get('_version',1)} — {rules.get('_update_reason','')}")
                return rules
            except Exception as e:
                logger.error(f"[LOGIC] Load error: {e}")
        return {}

    def _save(self, rules: dict):
        with open(LOGIC_FILE, "w") as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)
        self._append_history(rules)
        self._rules = rules
        logger.info(f"[LOGIC] Saved v{rules.get('_version',1)} — {rules.get('_update_reason','')}")

    def _append_history(self, rules: dict):
        history = []
        if os.path.exists(LOGIC_HISTORY_FILE):
            try:
                with open(LOGIC_HISTORY_FILE) as f:
                    history = json.load(f)
            except Exception:
                pass
        history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": rules.get("_version", 1),
            "reason": rules.get("_update_reason", ""),
            "snapshot": rules,
        })
        with open(LOGIC_HISTORY_FILE, "w") as f:
            json.dump(history[-20:], f, indent=2, ensure_ascii=False)

    def get(self, section: str) -> dict:
        return self._rules.get(section, {})

    def reload(self):
        self._rules = self._load()

    @property
    def rules(self) -> dict:
        return self._rules

    # ── Parameter accessors ─────────────────────────────

    @property
    def bos_h1_swing_left(self) -> int:
        return int(self.get("find_bos_h1").get("swing_left", 8))

    @property
    def bos_h1_swing_right(self) -> int:
        return int(self.get("find_bos_h1").get("swing_right", 8))

    @property
    def fvg_filter_by_bos(self) -> bool:
        return bool(self.get("find_fvg_h1").get("filter_by_bos", True))

    @property
    def fvg_min_gap_pct(self) -> float:
        return float(self.get("find_fvg_h1").get("min_gap_pct", 0.05))

    @property
    def idm_m5_gap_min(self) -> int:
        return int(self.get("find_idm_m5").get("bullish", {}).get("gap_min_candles", 1))

    @property
    def bos_m5_require_idm(self) -> bool:
        return bool(self.get("find_bos_m5").get("require_idm_first", True))

    @property
    def entry_skip_if_outside_fvg(self) -> bool:
        return bool(self.get("entry").get("skip_if_outside_fvg", True))

    @property
    def sl_buffer_pct(self) -> float:
        return float(self.get("stop_loss").get("buffer_pct", 0.0))

    @property
    def tp_min_rr(self) -> float:
        return float(self.get("take_profit").get("min_rr", 2.0))

    # ── AI Update on Loss ────────────────────────────────

    def ai_update_on_loss(self, client: Groq, model: str,
                          closed_trade: dict, debrief: dict) -> bool:
        """
        AI update logic rules setelah loss.
        Bisa mengubah kondisi/metode, bukan hanya angka.
        """
        rules_str = json.dumps(self._rules, indent=2, ensure_ascii=False)
        trade_str = (
            f"Direction: {closed_trade.get('direction','?')} | "
            f"Entry: {closed_trade.get('entry',0):.2f} | "
            f"SL: {closed_trade.get('sl',0):.2f} | "
            f"Exit: {closed_trade.get('exit_price',0):.2f} | "
            f"Setup: {closed_trade.get('setup','?')} | "
            f"Notes: {closed_trade.get('notes','')[:100]}"
        )
        culprit  = debrief.get("culprit", "unknown")
        root     = debrief.get("root_cause", "")
        new_rule = debrief.get("new_rule", "")

        prompt = f"""Kamu adalah logic optimizer untuk trading bot ICT. Ada trade yang loss.
Tugasmu: update LOGIKA pencarian (kondisi/metode), bukan hanya angka.

TRADE YANG LOSS:
{trade_str}

HASIL DEBRIEF:
Culprit: {culprit}
Root cause: {root}
Rule baru yang disarankan: {new_rule}

LOGIC RULES SAAT INI:
{rules_str}

TUGASMU:
Identifikasi LOGIKA mana yang menyebabkan loss dan update.
Contoh perubahan yang valid:
- find_bos_h1.swing_left: 8 → 10 (swing lebih ketat)
- find_fvg_h1.min_gap_pct: 0.05 → 0.1 (FVG minimum lebih besar)
- find_idm_m5.bullish.gap_min_candles: 1 → 2 (IDM lebih valid)
- stop_loss.buffer_pct: 0.0 → 0.05 (SL sedikit lebih longgar)
- entry.skip_if_outside_fvg: true → true (tetap, konfirmasi)

Jangan ubah yang tidak relevan. Perubahan harus incremental.
Jelaskan reasoning dalam _update_reason.

WAJIB: balas HANYA JSON murni — logic_rules lengkap yang sudah diupdate:
{{
  "_version": {self._rules.get('_version', 1) + 1},
  "_last_updated": "{datetime.now(timezone.utc).isoformat()}",
  "_update_reason": "penjelasan singkat apa yang diubah dan kenapa",
  "_update_count": {self._rules.get('_update_count', 0) + 1},
  ... (semua section lainnya)
}}"""

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=900,
            )
            raw = resp.choices[0].message.content.strip()

            import re
            parsed = None
            try:
                parsed = json.loads(raw)
            except Exception:
                m = re.search(r'(\{[\s\S]*\})', raw)
                if m:
                    try:
                        parsed = json.loads(m.group(1))
                    except Exception:
                        pass

            if not parsed:
                logger.warning(f"[LOGIC] AI response tidak bisa di-parse: {raw[:150]}")
                return False

            if parsed.get("_version", 0) <= self._rules.get("_version", 1):
                parsed["_version"] = self._rules.get("_version", 1) + 1

            # Restore key yang hilang
            for key in self._rules:
                if key not in parsed and not key.startswith("_"):
                    parsed[key] = self._rules[key]

            self._save(parsed)
            self._log_diff(self._rules, parsed)
            logger.info(f"[LOGIC] ✏️ Updated v{parsed.get('_version')} — {parsed.get('_update_reason','')}")
            return True

        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                logger.warning(f"[LOGIC] Rate limit: {err[:80]}")
            else:
                logger.error(f"[LOGIC] Error: {err[:100]}")
            return False

    def _log_diff(self, old: dict, new: dict):
        for key in new:
            if key.startswith("_"):
                continue
            if isinstance(new[key], dict) and isinstance(old.get(key), dict):
                for subkey, new_val in new[key].items():
                    if subkey == "comment":
                        continue
                    old_val = old.get(key, {}).get(subkey)
                    if old_val != new_val:
                        logger.info(f"[LOGIC DIFF] {key}.{subkey}: {old_val} → {new_val}")

    def get_context_for_ai(self) -> str:
        """Return full JSON logic_rules untuk AI — ini adalah otak cara kerja bot."""
        import json
        r = {k: v for k, v in self._rules.items() if not k.startswith("_")}
        return json.dumps(r, ensure_ascii=False, indent=None)
