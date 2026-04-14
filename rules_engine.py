"""
RulesEngine — Baca, apply, dan update rules trading.

Rules disimpan di data/rules.json.
Bot baca rules setiap siklus → semua parameter dinamis.
Setiap loss → AI review dan update rules yang salah.
"""

import os
import json
import logging
from datetime import datetime, timezone
from groq import Groq

logger = logging.getLogger(__name__)

RULES_FILE = "data/rules.json"
RULES_HISTORY_FILE = "data/rules_history.json"


class RulesEngine:
    def __init__(self):
        os.makedirs("data", exist_ok=True)
        self._rules = self._load()

    def _load(self) -> dict:
        if os.path.exists(RULES_FILE):
            try:
                with open(RULES_FILE) as f:
                    rules = json.load(f)
                logger.info(f"[RULES] Loaded v{rules.get('_version',1)} — {rules.get('_update_reason','')}")
                return rules
            except Exception as e:
                logger.error(f"[RULES] Load error: {e} — pakai default")
        return self._default()

    def _save(self, rules: dict):
        with open(RULES_FILE, "w") as f:
            json.dump(rules, f, indent=2, ensure_ascii=False)
        # Simpan history
        self._append_history(rules)
        self._rules = rules
        logger.info(f"[RULES] Saved v{rules.get('_version',1)} — {rules.get('_update_reason','')}")

    def _append_history(self, rules: dict):
        history = []
        if os.path.exists(RULES_HISTORY_FILE):
            try:
                with open(RULES_HISTORY_FILE) as f:
                    history = json.load(f)
            except Exception:
                pass
        history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "version": rules.get("_version", 1),
            "reason": rules.get("_update_reason", ""),
            "snapshot": rules,
        })
        # Simpan max 20 history
        with open(RULES_HISTORY_FILE, "w") as f:
            json.dump(history[-20:], f, indent=2, ensure_ascii=False)

    def get(self, *keys, default=None):
        """Akses nested rule. Contoh: get('idm_h1', 'gap_min_candles')"""
        val = self._rules
        for k in keys:
            if isinstance(val, dict):
                val = val.get(k)
            else:
                return default
        return val if val is not None else default

    @property
    def rules(self) -> dict:
        return self._rules

    def reload(self):
        """Reload dari file (kalau file diubah AI)."""
        self._rules = self._load()

    # ── Parameter accessors ─────────────────────────────

    @property
    def bos_h1_lookback(self) -> int:
        return int(self.get("bos_h1", "lookback_candles", default=40))

    @property
    def bos_h1_swing_min(self) -> int:
        return int(self.get("bos_h1", "swing_min_candles", default=1))

    @property
    def idm_h1_gap_min(self) -> int:
        return int(self.get("idm_h1", "gap_min_candles", default=1))

    @property
    def idm_h1_max_search(self) -> int:
        return int(self.get("idm_h1", "max_search_candles", default=10))

    @property
    def idm_m5_gap_min(self) -> int:
        return int(self.get("idm_m5", "gap_min_candles", default=1))

    @property
    def idm_m5_max_search(self) -> int:
        return int(self.get("idm_m5", "max_search_candles", default=10))

    @property
    def swing_range_min_pct(self) -> float:
        return float(self.get("swing_range", "min_range_pct", default=0.3))

    @property
    def ob_require_body_close(self) -> bool:
        return bool(self.get("ob", "require_body_close", default=True))

    @property
    def ob_buffer_pct(self) -> float:
        return float(self.get("ob", "sweep_fakeout_buffer_pct", default=0.0))

    @property
    def bos_m5_lookback(self) -> int:
        return int(self.get("bos_m5", "lookback_candles", default=15))

    @property
    def bos_m5_require_idm(self) -> bool:
        return bool(self.get("bos_m5", "require_idm_first", default=True))

    @property
    def entry_min_confidence(self) -> float:
        return float(self.get("entry", "min_confidence", default=0.6))

    @property
    def entry_allowed_setups(self) -> list:
        return self.get("entry", "allowed_setups",
                        default=["Quasimodo", "RBS", "OB_retest", "MSNR_support"])

    @property
    def sl_buffer_pct(self) -> float:
        return float(self.get("sl", "buffer_pct", default=0.0))

    @property
    def tp_min_rr(self) -> float:
        return float(self.get("tp", "min_rr", default=1.5))

    # ── AI Update ───────────────────────────────────────

    def ai_update_on_loss(self, client: Groq, model: str, closed_trade: dict,
                          debrief: dict, current_price: float) -> bool:
        """
        Panggil AI untuk review rules setelah loss.
        AI menganalisis trade yang loss + debrief → update rules yang relevan.
        Return True kalau rules berubah.
        """
        rules_str = json.dumps(self._rules, indent=2, ensure_ascii=False)
        trade_str = (
            f"Direction: {closed_trade.get('direction','?')} | "
            f"Entry: {closed_trade.get('entry',0):.2f} | "
            f"SL: {closed_trade.get('sl',0):.2f} | "
            f"Exit: {closed_trade.get('exit_price',0):.2f} | "
            f"PnL: {closed_trade.get('pnl',0):.4f} | "
            f"Setup: {closed_trade.get('setup','?')} | "
            f"Notes: {closed_trade.get('notes','')[:100]}"
        )
        culprit  = debrief.get("culprit", "unknown")
        root     = debrief.get("root_cause", "")
        new_rule = debrief.get("new_rule", "")

        prompt = f"""Kamu adalah rules optimizer untuk trading bot ICT. Ada trade yang loss — tugasmu update rules bot supaya kesalahan ini tidak terulang.

TRADE YANG LOSS:
{trade_str}

HASIL DEBRIEF 4 AI:
Culprit: {culprit}
Root cause: {root}
Rule yang disarankan: {new_rule}

RULES BOT SAAT INI:
{rules_str}

TUGASMU:
1. Identifikasi parameter mana yang menyebabkan loss ini
2. Update HANYA parameter yang relevan — jangan ubah yang tidak berkaitan
3. Perubahan harus incremental — jangan langsung ekstrem
4. Jelaskan kenapa dalam _update_reason

Contoh perubahan incremental yang baik:
- idm_h1.gap_min_candles: 1 → 2 (kalau IDM terlalu agresif)
- bos_h1.lookback_candles: 40 → 50 (kalau BOS missed karena lookback kurang)
- entry.min_confidence: 0.6 → 0.65 (kalau entry terlalu mudah)
- sl.buffer_pct: 0.0 → 0.1 (kalau SL terlalu mepet)

WAJIB: balas HANYA JSON murni — rules lengkap yang sudah diupdate:
{{
  "_version": {self._rules.get('_version', 1) + 1},
  "_last_updated": "{datetime.now(timezone.utc).isoformat()}",
  "_update_reason": "penjelasan singkat apa yang diubah dan kenapa",
  "_update_count": {self._rules.get('_update_count', 0) + 1},
  ... (semua rules lainnya, ubah hanya yang perlu)
}}"""

        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=800,
            )
            raw = resp.choices[0].message.content.strip()

            # Parse JSON
            import re
            parsed = None
            try:
                parsed = json.loads(raw)
            except Exception:
                match = re.search(r'(\{[\s\S]*\})', raw)
                if match:
                    try:
                        parsed = json.loads(match.group(1))
                    except Exception:
                        pass

            if not parsed:
                logger.warning(f"[RULES] AI response tidak bisa di-parse: {raw[:150]}")
                return False

            # Validasi: harus ada _version yang lebih tinggi
            if parsed.get("_version", 0) <= self._rules.get("_version", 1):
                parsed["_version"] = self._rules.get("_version", 1) + 1

            # Pastikan semua key lama masih ada (AI tidak menghapus key penting)
            for key in self._rules:
                if key not in parsed and not key.startswith("_"):
                    parsed[key] = self._rules[key]  # Restore key yang hilang

            self._save(parsed)

            # Log perubahan
            reason = parsed.get("_update_reason", "")
            logger.info(f"[RULES] ✏️ Updated v{parsed.get('_version')} — {reason}")

            # Log diff
            self._log_diff(self._rules, parsed)
            return True

        except Exception as e:
            err = str(e)
            if "429" in err or "rate_limit" in err.lower():
                logger.warning(f"[RULES] Rate limit saat update rules: {err[:80]}")
            else:
                logger.error(f"[RULES] Error update rules: {err[:100]}")
            return False

    def _log_diff(self, old: dict, new: dict):
        """Log perubahan spesifik antara rules lama dan baru."""
        for key in new:
            if key.startswith("_"):
                continue
            if isinstance(new[key], dict) and isinstance(old.get(key), dict):
                for subkey in new[key]:
                    if subkey == "comment":
                        continue
                    old_val = old.get(key, {}).get(subkey)
                    new_val = new[key].get(subkey)
                    if old_val != new_val:
                        logger.info(f"[RULES DIFF] {key}.{subkey}: {old_val} → {new_val}")

    def _default(self) -> dict:
        return {
            "_version": 1, "_last_updated": None,
            "_update_reason": "Default", "_update_count": 0,
            "bos_h1": {"lookback_candles": 40, "swing_min_candles": 1},
            "idm_h1": {"gap_min_candles": 1, "max_search_candles": 10},
            "idm_m5": {"gap_min_candles": 1, "max_search_candles": 10},
            "swing_range": {"min_range_pct": 0.3},
            "ob": {"sweep_fakeout_buffer_pct": 0.0, "require_body_close": True},
            "bos_m5": {"lookback_candles": 15, "require_idm_first": True},
            "mss_m5": {"enabled": True},
            "entry": {"min_confidence": 0.6, "prefer_setup": "Quasimodo",
                      "allowed_setups": ["Quasimodo", "RBS", "OB_retest", "MSNR_support"]},
            "sl": {"use_msnr": True, "buffer_pct": 0.0},
            "tp": {"min_rr": 1.5, "target": "next_liquidity"},
            "filter": {"skip_if_ranging": True, "ranging_threshold_pct": 0.5},
        }
