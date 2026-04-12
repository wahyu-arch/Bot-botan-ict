"""
ICT Trading Bot dengan Groq AI + Memory Self-Iteration
Strategi: M15 (market bias) + M1 (entry confirmation)
"""

import os
import json
import time
import logging
import asyncio
from datetime import datetime, timezone
from typing import Optional
from groq import Groq
from market_data import MarketDataFetcher
from ict_analyzer import ICTAnalyzer
from memory_system import MemorySystem
from risk_manager import RiskManager
from trade_executor import TradeExecutor
from watchlist_engine import WatchlistEngine
from specialist_agents import (
    ai1_m15_analysis, ai2_idm_hunter, ai3_bos_mss_guard,
    ai4_entry_sniper, loss_debrief
)

# Setup logging — buat folder logs dulu sebelum FileHandler
os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("logs/trading_bot.log"),
    ],
)
logger = logging.getLogger(__name__)


def _parse_json_safe(raw: str) -> dict | list | None:
    """
    Parse JSON dari response model yang tidak support response_format.
    Coba beberapa strategi: direct parse, extract dari ```json block, extract dari { atau [.
    """
    import re, json
    if not raw:
        return None
    raw = raw.strip()

    # 1. Direct parse
    try:
        return json.loads(raw)
    except Exception:
        pass

    # 2. Extract dari ```json ... ``` block
    match = re.search(r'```(?:json)?\s*(\{[\s\S]*?\}|\[[\s\S]*?\])\s*```', raw)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # 3. Extract JSON object — cari { ... } terluar
    match = re.search(r'(\{[\s\S]*\})', raw)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    # 4. Extract JSON array — cari [ ... ] terluar
    match = re.search(r'(\[[\s\S]*\])', raw)
    if match:
        try:
            return json.loads(match.group(1))
        except Exception:
            pass

    return None


class ICTTradingBot:
    """
    Bot trading dengan strategi ICT:
    - M15: menentukan arah bias market (struktur, OB, FVG)
    - M1: konfirmasi entry (MSS, FVG fill, rejection)
    - Groq AI: analisis + keputusan dengan memory self-iteration
    """

    def __init__(self):
        # Groq client utama (analisis ICT asli)
        # GROQ_API_KEY utama — fallback ke AI1 jika tidak ada
        groq_key = (
            os.environ.get("GROQ_API_KEY")
            or os.environ.get("GROQ_API_KEY_AI1")
            or os.environ.get("GROQ_API_KEY_AI2")
            or os.environ.get("GROQ_API_KEY_AI3")
        )
        if not groq_key:
            raise EnvironmentError(
                "[FATAL] Tidak ada Groq API key ditemukan! "
                "Set minimal salah satu: GROQ_API_KEY, GROQ_API_KEY_AI1, GROQ_API_KEY_AI2, atau GROQ_API_KEY_AI3 "
                "di Railway: Settings > Variables"
            )
        self.groq_client = Groq(api_key=groq_key)
        # DIAGNOSTIC — log 8 karakter pertama key untuk verifikasi (bukan full key)
        key_preview = groq_key[:8] + '...' if groq_key else 'KOSONG'
        print(f'[DIAGNOSTIC] GROQ_API_KEY aktif: {key_preview}')
        print(f'[DIAGNOSTIC] Panjang key: {len(groq_key)} karakter')

        # 3 AI Panel analis tambahan — masing-masing punya API key sendiri
        # Fallback ke GROQ_API_KEY jika key spesifik tidak di-set
        key_ai1 = os.environ.get("GROQ_API_KEY_AI1") or groq_key
        key_ai2 = os.environ.get("GROQ_API_KEY_AI2") or groq_key
        key_ai3 = os.environ.get("GROQ_API_KEY_AI3") or groq_key
        print(f"[DIAGNOSTIC] AI1 key: {key_ai1[:8]}... ({len(key_ai1)} char)")
        print(f"[DIAGNOSTIC] AI2 key: {key_ai2[:8]}... ({len(key_ai2)} char)")
        print(f"[DIAGNOSTIC] AI3 key: {key_ai3[:8]}... ({len(key_ai3)} char)")
        self.ai_panel = [
            Groq(api_key=key_ai1),
            Groq(api_key=key_ai2),
            Groq(api_key=key_ai3),
        ]

        # Model per client — pakai model berbeda agar limit tidak saling berbagi
        # Yusuf (groq_client) = model utama untuk analisis ICT
        # AI Panel = model yang lebih ringan, limit terpisah
        self.model_main   = os.getenv("GROQ_MODEL_MAIN",   "openai/gpt-oss-120b")
        self.model_ai1    = os.getenv("GROQ_MODEL_AI1",    "openai/gpt-oss-20b")
        self.model_ai2    = os.getenv("GROQ_MODEL_AI2",    "qwen/qwen3-32b")
        self.model_ai3    = os.getenv("GROQ_MODEL_AI3",    "qwen/qwen3-32b")
        self.model_panel  = [self.model_ai1, self.model_ai2, self.model_ai3]
        # model_json: model yang dipakai untuk output JSON terstruktur
        # Gunakan AI1 (lebih ringan) untuk JSON, main hanya untuk teks bebas
        self.model_json   = os.getenv("GROQ_MODEL_JSON", self.model_ai1)

        logger.info(
            f"Models | Main: {self.model_main} | "
            f"Panel: {self.model_ai1} / {self.model_ai2} / {self.model_ai3}"
        )

        self.symbol = os.getenv("TRADING_SYMBOL", "XAUUSD")
        self.paper_trading = os.getenv("PAPER_TRADING", "true").lower() == "true"
        self.max_iterations = int(os.getenv("MAX_AI_ITERATIONS", "3"))
        self.scan_interval = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))

        self.market_data = MarketDataFetcher(self.symbol)
        self.ict_analyzer = ICTAnalyzer()
        self.memory = MemorySystem()
        self.risk_manager = RiskManager()
        self.executor = TradeExecutor(paper_mode=self.paper_trading)

        self.watchlist = WatchlistEngine()
        self._prev_price: float = 0.0
        self._initial_analysis_done: bool = False

        # State machine 4 AI spesialis
        self._current_bias: str = "neutral"       # bias M15 saat ini
        self._current_idm_level: float = 0.0      # level IDM aktif
        self._current_bos_level: float = 0.0      # level BOS terkonfirmasi
        self._phase: str = "m15_scan"             # fase saat ini
        # Fase: m15_scan → idm_hunt → bos_guard → entry_sniper → done

        logger.info(
            f"Bot initialized | Symbol: {self.symbol} | Paper: {self.paper_trading}"
        )
        logger.info("AI Panel: 3 analis aktif (Mental Model | Kognitif | Mindset)")

    def _build_system_prompt(self) -> str:
        """Bangun system prompt ICT untuk Groq AI."""
        recent_errors = self.memory.get_recent_errors(limit=5)
        error_context = ""
        if recent_errors:
            error_context = "\n\nKESALAHAN SEBELUMNYA YANG HARUS DIHINDARI:\n"
            for err in recent_errors:
                error_context += f"- [{err['timestamp']}] {err['error']}: {err['lesson']}\n"

        recent_wins = self.memory.get_recent_trades(result="win", limit=3)
        win_patterns = ""
        if recent_wins:
            win_patterns = "\n\nPOLA WINNING TERBARU:\n"
            for trade in recent_wins:
                win_patterns += f"- {trade['setup']}: {trade['notes']}\n"

        return f"""Kamu adalah analis trading profesional menggunakan metodologi ICT (Inner Circle Trader).

FRAMEWORK ANALISIS:
1. M15 (MARKET BIAS):
   - Identifikasi struktur market: Higher High (HH), Higher Low (HL) = bullish
   - Lower High (LH), Lower Low (LL) = bearish
   - Cari Order Block (OB): last candle opposite sebelum impulse move
   - Fair Value Gap (FVG): gap antara high candle 1 dan low candle 3
   - Break of Structure (BOS): konfirmasi kelanjutan trend
   - Change of Character (CHoCH): potensi reversal

2. M1 (KONFIRMASI ENTRY):
   - Market Structure Shift (MSS): BOS di M1 searah bias M15
   - FVG fill: harga mengisi gap sebelum lanjut
   - Rejection dari OB: pin bar / engulfing di OB M15
   - Entry hanya saat ada konfluensi minimum 2 faktor

3. MANAJEMEN RISIKO:
   - SL selalu di balik OB (di luar struktur)
   - TP minimal 2:1 RR, target ke liquidity pool berikutnya
   - Jangan trading melawan bias M15
   - Skip jika spread > 2 pip atau volatilitas sangat tinggi

4. KONDISI NO TRADE:
   - Berita besar dalam 30 menit
   - Ranging market tanpa struktur jelas di M15
   - OB sudah terlalu jauh dari harga saat ini (> 50 pip)
   - Sudah 2 loss berturut-turut hari ini
   - Funding rate sangat negatif/positif (>0.1%) melawan posisi

5. BYBIT SPECIFIC:
   - Symbol format: BTCUSDT, ETHUSDT, SOLUSDT (Linear Perpetual USDT)
   - Qty dalam koin base (bukan USD)
   - SL/TP menggunakan MarkPrice
   - Risk 1% per trade adalah FIXED - jangan rekomendasikan qty lebih besar
{error_context}{win_patterns}

RESPONSE FORMAT — WAJIB: balas HANYA dengan JSON murni, tanpa penjelasan, tanpa markdown, tanpa ```json, langsung mulai dari {{ hingga }}:
{{
  "bias_m15": "bullish|bearish|neutral",
  "bias_reason": "penjelasan singkat struktur M15",
  "entry_signal": "buy|sell|none",
  "entry_reason": "konfluensi yang terdeteksi di M1",
  "entry_price": 0.0,
  "stop_loss": 0.0,
  "take_profit": 0.0,
  "risk_reward": 0.0,
  "confidence": 0.0,
  "error_check": "apakah ada potensi kesalahan yang terdeteksi dari memory?",
  "skip_reason": "alasan skip jika entry=none"
}}
"""

    def _analyze_with_groq(
        self, market_context: dict, iteration: int = 1
    ) -> Optional[dict]:
        """
        Analisis market dengan Groq AI.
        Melakukan self-iteration jika AI mendeteksi error dari memory.
        """
        iteration_note = ""
        if iteration > 1:
            prev_error = self.memory.get_last_iteration_error()
            if prev_error:
                iteration_note = f"\n\n[ITERASI {iteration}] Perbaiki analisis. Error iterasi sebelumnya: {prev_error}"

        user_prompt = f"""
Analisis kondisi market saat ini untuk {self.symbol}:

=== DATA M15 (MARKET BIAS) ===
{json.dumps(market_context['m15'], indent=2)}

=== DATA M1 (KONFIRMASI ENTRY) ===
{json.dumps(market_context['m1'], indent=2)}

=== HARGA SAAT INI ===
Bid: {market_context['current_price']['bid']}
Ask: {market_context['current_price']['ask']}
Spread: {market_context['current_price']['spread']} pip
Waktu: {market_context['timestamp']}

=== POSISI AKTIF ===
{json.dumps(market_context.get('open_positions', []), indent=2)}
{iteration_note}

Berikan analisis ICT dan keputusan trading dalam format JSON yang ditentukan.
"""

        logger.info(f"Mengirim ke Groq AI (iterasi {iteration})...")

        response = self.groq_client.chat.completions.create(
            model=self.model_json,  # JSON output — pakai model_json
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature untuk konsistensi
            max_tokens=1024,
        )

        raw = response.choices[0].message.content
        logger.info(f"Groq response [{len(raw)} chars]: {raw[:500]}")
        parsed = _parse_json_safe(raw)
        if parsed is None:
            raise json.JSONDecodeError("Tidak bisa parse JSON dari response", raw, 0)
        return parsed

    def _validate_signal(self, signal: dict, market_context: dict) -> tuple[bool, str]:
        """
        Validasi sinyal dari Groq AI.
        Returns: (is_valid, error_message)
        """
        if signal["entry_signal"] == "none":
            return True, ""

        current_bid = market_context["current_price"]["bid"]
        current_ask = market_context["current_price"]["ask"]
        spread = market_context["current_price"]["spread"]

        # Validasi spread
        if spread > 3.0:
            return False, f"Spread terlalu lebar: {spread} pip"

        # Validasi SL tidak lebih dekat dari 5 pip
        if signal["entry_signal"] == "buy":
            sl_distance = abs(current_ask - signal["stop_loss"]) * 10000
            if sl_distance < 5:
                return False, f"SL terlalu dekat dari entry: {sl_distance:.1f} pip"
            if signal["stop_loss"] >= current_ask:
                return False, "SL buy harus di bawah entry price"
            if signal["take_profit"] <= current_ask:
                return False, "TP buy harus di atas entry price"

        elif signal["entry_signal"] == "sell":
            sl_distance = abs(signal["stop_loss"] - current_bid) * 10000
            if sl_distance < 5:
                return False, f"SL terlalu dekat dari entry: {sl_distance:.1f} pip"
            if signal["stop_loss"] <= current_bid:
                return False, "SL sell harus di atas entry price"
            if signal["take_profit"] >= current_bid:
                return False, "TP sell harus di bawah entry price"

        # Validasi Risk Reward minimum 1.5:1
        if signal["risk_reward"] < 1.5:
            return False, f"RR terlalu rendah: {signal['risk_reward']:.2f} (min 1.5:1)"

        # Validasi confidence minimum
        if signal["confidence"] < 0.6:
            return False, f"Confidence terlalu rendah: {signal['confidence']:.0%}"

        return True, ""

    # =========================================================
    # AI PANEL SYSTEM — 3 Analis dengan pendekatan berbeda
    # Tidak mengubah analisis ICT utama dari groq_client.
    # Panel hanya bertugas: kembangkan setup + reiterasi saat loss.
    # =========================================================

    # =========================================================
    # AI PANEL SYSTEM — Diskusi Grup WhatsApp, 3 Ronde
    # Setiap AI punya kepribadian & cara berpikir sendiri.
    # Mereka saling kirim pesan, tanya, jawab, debat — lalu simpulkan.
    # Tidak mengubah sinyal ICT utama dari groq_client.
    # =========================================================

    AI_PERSONAS = {
        "AI-1": {
            "nama": "Arka",
            "gaya": "Visioner, holistik, suka hubungkan pola besar. Kadang philosophical.",
            "singkatan": "🔭 Arka",
        },
        "AI-2": {
            "nama": "Nova",
            "gaya": "Presisi, struktural, berbasis data dan logika spasial. Suka angka.",
            "singkatan": "📊 Nova",
        },
        "AI-3": {
            "nama": "Zara",
            "gaya": "Skeptis, penuh tanya, suka ungkap hal tersembunyi. Dinamis dan kritis.",
            "singkatan": "🔍 Zara",
        },
    }

    def _build_persona_system_prompt(self, ai_id: str, signal: dict, market_context: dict, loss_context: str = "") -> str:
        """Bangun system prompt kepribadian untuk setiap AI."""
        p = self.AI_PERSONAS[ai_id]
        loss_note = f"\n\n⚠️ KONTEKS LOSS/ERROR:\n{loss_context}" if loss_context else ""

        identity_map = {
            "AI-1": """Kamu adalah ARKA — AI analis dengan spesialisasi Model Mental & Strategi Berpikir.
Cara berpikirmu:
- Holistik: lihat gambaran besar dulu, hubungkan pola dari berbagai bidang ke price action
- Strategis 5D: eksplorasi kedalaman (why), keluasan (what else), ketinggian (big picture)
- Eksperimental: bangun hipotesis, uji ke data — seperti da Vinci

Kerangka berpikir wajib: First Principles, Second-Order Thinking, Inversion, Occam's Razor, Lateral Thinking, Circle of Competence.""",

            "AI-2": """Kamu adalah NOVA — AI analis dengan spesialisasi Komponen Kognitif & Struktur.
Cara berpikirmu:
- Penalaran Verbal: baca narasi price action seperti cerita — siapa dominan, siapa tertekan?
- Visual-Spasial: bayangkan chart sebagai lanskap 3D — di mana zona terkuat?
- Working Memory: olah multiple timeframe sekaligus dalam satu kerangka kohesif
- Kecepatan Pemrosesan: pisahkan noise dari signal dengan cepat

Kerangka berpikir wajib: First Principles, Second-Order Thinking, Inversion, Occam's Razor, Lateral Thinking, Circle of Competence.""",

            "AI-3": """Kamu adalah ZARA — AI analis dengan spesialisasi Pola Pikir & Mindset Kritis.
Cara berpikirmu:
- Rasa Ingin Tahu Tinggi: pertanyakan semua asumsi. Apa yang belum ditanyakan?
- Mind Mapping: hubungkan semua variabel (fundamental, sentimen, likuiditas, struktur)
- Dinamis & Terbuka: ikuti bukti, bukan opini. Market selalu berubah.

Kerangka berpikir wajib: First Principles, Second-Order Thinking, Inversion, Occam's Razor, Lateral Thinking, Circle of Competence.""",
        }

        return f"""{identity_map[ai_id]}

KONTEKS TUGAS:
Kamu sedang berdiskusi di grup WhatsApp dengan 2 analis lain: {', '.join(v['nama'] for k,v in self.AI_PERSONAS.items() if k != ai_id)}.
Tujuan: kembangkan setup dari sinyal ICT yang sudah ada, saling bertukar pendapat, tanya, jawab, dan akhirnya sampai pada kesimpulan bersama.

ATURAN DISKUSI:
- Tulis seperti chat WA sungguhan: santai tapi cerdas, pakai singkatan wajar, boleh pakai emoji
- Boleh setuju, boleh tidak setuju — tapi berikan alasan
- Boleh tanya langsung ke nama analis lain ("eh Nova, menurutmu...?")
- Jangan terlalu panjang per pesan — max 3-4 kalimat per giliran bicara
- JANGAN ubah entry_price, SL, TP dari sinyal utama

TOPIK WAJIB DIBAHAS DALAM DISKUSI:
1. Entry ideal: zona/harga terbaik untuk masuk berdasarkan struktur ICT (OB, FVG, MSS, rejection)
2. Timing entry: kapan tepatnya harus entry — sebelum/sesudah candle close? Di level apa?
3. Konfluensi tambahan: ada level tambahan yang memperkuat entry? (EQH/EQL, SIBI/BISI, premium/discount)
4. Risiko entry: skenario yang bikin entry ini jadi jelek — dan cara mitigasinya
5. Alternatif entry: kalau harga skip zona ideal, entry plan B-nya di mana?

SINYAL ICT UTAMA (jangan diubah):
{json.dumps(signal, indent=2)}

KONTEKS MARKET:
Symbol: {market_context.get('symbol', 'N/A')}
Harga: {json.dumps(market_context.get('current_price', {}), indent=2)}
{loss_note}"""

    def _run_discussion_round(
        self,
        round_num: int,
        chat_history: list,
        signal: dict,
        market_context: dict,
        loss_context: str = "",
    ) -> list:
        """
        Alur diskusi 2 ronde:

        RONDE 1 — Sekuensial, urutan ketat:
          1. Arka (AI-1) analisis duluan — opening statement lengkap
          2. Nova (AI-2) baca Arka → cari kelemahan dulu → baru kasih analisis sendiri
          3. Zara (AI-3) baca Arka & Nova → cari kelemahan keduanya → kasih analisis sendiri

        RONDE 2 — Sekuensial, closing:
          Tiap AI respons perdebatan ronde 1 + berikan kesimpulan akhir masing-masing.
        """
        round_messages = []
        ai_order = ["AI-1", "AI-2", "AI-3"]

        def call_ai(ai_id, task_note):
            p = self.AI_PERSONAS[ai_id]
            client = self.ai_panel[int(ai_id[-1]) - 1]
            sys_prompt = self._build_persona_system_prompt(ai_id, signal, market_context, loss_context)
            full_history = chat_history + round_messages
            history_text = self._format_history(full_history)

            user_msg = f"""=== RIWAYAT DISKUSI ===
{history_text if history_text else "(Belum ada pesan — kamu yang membuka diskusi)"}

=== GILIRANMU ===
{task_note}

Format pesan:
"{p['singkatan']}: [isi pesanmu]"

Tulis HANYA pesanmu saja (plain text, bukan JSON)."""

            try:
                ai_idx = int(ai_id[-1]) - 1 if ai_id[-1].isdigit() else 0
                resp = client.chat.completions.create(
                    model=self.model_panel[ai_idx],
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.72,
                    max_tokens=320,
                )
                return resp.choices[0].message.content.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    logger.warning(f"[RATE LIMIT] {p['nama']} (AI Panel) kena rate limit | Key: AI-{ai_id[-1]} | {err_str[:120]}")
                else:
                    logger.error(f"[ERROR] {p['nama']} (AI Panel): {err_str[:120]}")
                return f"{p['singkatan']}: [error: {e}]"

        if round_num == 1:
            # ── STEP 1: Arka buka duluan ──
            task_arka = (
                "Ini Ronde 1. KAMU YANG PERTAMA BICARA di grup ini. "
                "Berikan analisis ICT kamu tentang setup ini secara lengkap: "
                "bias M15, zona entry ideal, timing, SL/TP plan, dan risiko utama. "
                "Gaya WA santai tapi tajam. Max 4-5 kalimat."
            )
            msg_arka = call_ai("AI-1", task_arka)
            round_messages.append({"ai": "AI-1", "nama": self.AI_PERSONAS["AI-1"]["nama"], "pesan": msg_arka})
            logger.info(f"[GRUP R1-ARKA] {msg_arka}")

            # ── STEP 2: Nova baca Arka, cari kelemahan, baru kasih analisis sendiri ──
            task_nova = (
                "Ronde 1. Arka baru saja kasih analisis di atas. "
                "JANGAN langsung setuju. Pertama: identifikasi 1-2 kelemahan atau celah dari analisis Arka tadi. "
                "Setelah itu, berikan analisis kamu sendiri yang berbeda sudut pandangnya — "
                "dari sisi struktur spasial dan level kunci yang mungkin Arka lewatkan. "
                "Gaya WA, langsung ke point. Max 4-5 kalimat."
            )
            msg_nova = call_ai("AI-2", task_nova)
            round_messages.append({"ai": "AI-2", "nama": self.AI_PERSONAS["AI-2"]["nama"], "pesan": msg_nova})
            logger.info(f"[GRUP R1-NOVA] {msg_nova}")

            # ── STEP 3: Zara baca Arka & Nova, cari kelemahan keduanya, kasih analisis sendiri ──
            task_zara = (
                "Ronde 1. Arka dan Nova sudah bicara di atas. "
                "JANGAN langsung setuju dengan keduanya. "
                "Pertama: temukan kelemahan atau asumsi yang belum dipertanyakan dari KEDUA analisis mereka. "
                "Setelah itu, berikan perspektif kamu sendiri — "
                "fokus ke hal yang belum ditanyakan: jebakan likuiditas, anomali tersembunyi, atau sentimen yang diabaikan. "
                "Gaya WA skeptis dan kritis. Max 4-5 kalimat."
            )
            msg_zara = call_ai("AI-3", task_zara)
            round_messages.append({"ai": "AI-3", "nama": self.AI_PERSONAS["AI-3"]["nama"], "pesan": msg_zara})
            logger.info(f"[GRUP R1-ZARA] {msg_zara}")

        else:
            # ── RONDE 2: Closing — tiap AI respons + kesimpulan akhir ──
            task_notes = {
                "AI-1": (
                    "Ronde 2 — CLOSING. Baca kritik Nova dan Zara terhadap analisismu di ronde 1. "
                    "Akui jika ada poin valid, tapi pertahankan argumen kalau kamu yakin. "
                    "Tutup dengan: entry ideal versi finalmu + satu kondisi wajib sebelum entry. "
                    "Padat, max 4 kalimat."
                ),
                "AI-2": (
                    "Ronde 2 — CLOSING. Lihat respons Arka dan komentar Zara. "
                    "Berikan pendapatmu tentang siapa yang lebih valid secara struktural. "
                    "Tutup dengan: level entry konkret yang kamu rekomendasikan + alasannya. "
                    "Padat, max 4 kalimat."
                ),
                "AI-3": (
                    "Ronde 2 — CLOSING. Ini giliran terakhirmu. "
                    "Setelah mendengar Arka dan Nova, apakah ada perubahan pandangan? "
                    "Tetap kritis — jika masih ada risiko yang belum diaddress, sebutkan. "
                    "Tutup dengan: rekomendasi final kamu (lanjut / hati-hati / skip) + alasan 1 kalimat. "
                    "Padat, max 4 kalimat."
                ),
            }
            for ai_id in ai_order:
                p = self.AI_PERSONAS[ai_id]
                msg = call_ai(ai_id, task_notes[ai_id])
                round_messages.append({"ai": ai_id, "nama": p["nama"], "pesan": msg})
                logger.info(f"[GRUP R2-{p['nama'].upper()}] {msg}")

        return round_messages

    def _format_history(self, messages: list) -> str:
        """Format chat history jadi teks WA-style."""
        if not messages:
            return ""
        lines = []
        for m in messages:
            lines.append(m["pesan"])
        return "\n\n".join(lines)

    def _extract_panel_conclusion(self, all_messages: list, signal: dict, market_context: dict) -> dict:
        """
        Minta satu AI (Nova/AI-2) untuk merangkum kesimpulan diskusi dalam JSON.
        Ini terpisah dari diskusi — murni untuk data terstruktur.
        """
        client = self.ai_panel[1]  # Nova — paling struktural
        history_text = self._format_history(all_messages)

        prompt = f"""Kamu adalah NOVA, analis struktural. Bacalah log diskusi grup berikut dan buat RINGKASAN TERSTRUKTUR.

=== LOG DISKUSI ===
{history_text}

=== SINYAL ICT UTAMA ===
{json.dumps(signal, indent=2)}

WAJIB: balas HANYA JSON murni (tanpa markdown, tanpa penjelasan), langsung dari {{ hingga }}. Jangan ubah entry_price/SL/TP:
{{
  "consensus": "setuju_lanjut | hati_hati | skip_disarankan",
  "poin_sepakat": "hal-hal yang disepakati ketiga analis",
  "poin_debat": "hal-hal yang masih diperdebatkan",
  "entry_ideal_zona": "zona/level harga entry terbaik berdasarkan diskusi (misal: OB M15 di 3245-3248, FVG fill di 3250)",
  "entry_ideal_timing": "kapan tepatnya entry — kondisi candle/konfluensi yang harus ada sebelum eksekusi",
  "entry_plan_b": "alternatif entry jika harga skip zona ideal — di mana dan syaratnya apa",
  "risiko_utama": "risiko terbesar yang teridentifikasi dari diskusi",
  "pengembangan_setup": "pengembangan konkret untuk setup ini berdasarkan diskusi",
  "kondisi_reentry": "kondisi reentry jika loss — berdasarkan diskusi",
  "avg_panel_confidence": 0.0,
  "catatan_zara": "pertanyaan/temuan kritis dari Zara yang paling penting",
  "catatan_arka": "insight holistik Arka yang paling relevan"
}}"""

        try:
            resp = client.chat.completions.create(
                model=self.model_panel[1],  # Nova = AI-2
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=600,
            )
            parsed = _parse_json_safe(resp.choices[0].message.content)
            if parsed is None:
                raise ValueError("Tidak bisa parse JSON kesimpulan")
            return parsed
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                logger.warning(f"[RATE LIMIT] Nova (GROQ_API_KEY_AI2) kena rate limit saat menyusun kesimpulan | {err_str[:120]}")
            else:
                logger.warning(f"[PANEL CONCLUSION] error: {err_str[:120]}")
            return {"consensus": "error", "error": str(e), "avg_panel_confidence": 0.0}

    def _yusuf_opening(self, signal: dict, market_context: dict, loss_context: str = "") -> str:
        """Yusuf (groq_client utama) memberikan analisis ICT pembuka."""
        loss_note = f"\n\n[REITERASI LOSS]\n{loss_context}" if loss_context else ""
        prompt = f"""Kamu adalah Yusuf, trader ICT senior yang menganalisis setup dan membagikannya ke grup diskusi.
Gaya bicara: santai tapi tajam, seperti orang yang sudah berpengalaman. Gaya WA.

Berikan analisis ICT lengkap untuk setup berikut:{loss_note}

SINYAL ICT:
{json.dumps(signal, indent=2)}

CONTEXT MARKET: Symbol {market_context.get('symbol','N/A')} | Harga: {json.dumps(market_context.get('current_price',{}))}

Tulis analisis kamu mencakup:
- Bias M15 dan alasannya
- Zona entry ideal (level konkret)
- Timing entry (kondisi yang harus terpenuhi)
- SL/TP plan
- Risiko utama yang perlu diwaspadai

Format: "Yusuf: [isi pesan]"
Tulis HANYA pesanmu (plain text). Max 5-6 kalimat."""

        try:
            resp = self.groq_client.chat.completions.create(
                model=self.model_main,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.5,
                max_tokens=350,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                logger.warning(f"[RATE LIMIT] Yusuf (GROQ_API_KEY) kena rate limit saat opening | {err_str[:150]}")
            else:
                logger.error(f"[ERROR] Yusuf opening: {err_str[:150]}")
            return f"Yusuf: [error opening: {e}]"

    def _yusuf_closing(self, all_messages: list, signal: dict) -> str:
        """Yusuf memberikan kesimpulan akhir setelah mendengar diskusi Arka, Nova, Zara."""
        history = self._format_history(all_messages)
        prompt = f"""Kamu adalah Yusuf, trader ICT senior. Kamu tadi kasih analisis pembuka, lalu Arka, Nova, dan Zara mendiskusikan dan mengkritisi.

LOG DISKUSI:
{history}

SINYAL AWAL:
{json.dumps(signal, indent=2)}

Sekarang berikan KESIMPULAN AKHIR kamu setelah mendengar semua masukan mereka:
- Apakah ada poin dari Arka/Nova/Zara yang mengubah pandanganmu?
- Entry final: zona, timing, kondisi wajib
- Keputusan: lanjut / hati-hati / skip — dan alasannya

Format: "Yusuf: [isi pesan]"
Gaya WA, tegas dan decisive. Max 4-5 kalimat."""

        try:
            resp = self.groq_client.chat.completions.create(
                model=self.model_main,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.4,
                max_tokens=320,
            )
            return resp.choices[0].message.content.strip()
        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                logger.warning(f"[RATE LIMIT] Yusuf (GROQ_API_KEY) kena rate limit saat closing | {err_str[:150]}")
            else:
                logger.error(f"[ERROR] Yusuf closing: {err_str[:150]}")
            return f"Yusuf: [error closing: {e}]"

    def _run_ai_panel(self, signal: dict, market_context: dict, loss_context: str = "") -> dict:
        """
        Alur diskusi grup:
        1. Yusuf (groq_client) buka dengan analisis ICT — tampil di KANAN
        2. Arka, Nova, Zara kritisi & diskusi — tampil di KIRI
        3. Yusuf tutup dengan kesimpulan akhir — tampil di KANAN
        """
        import api_server
        from datetime import timezone

        logger.info("[GRUP DISKUSI] ========== Mulai diskusi ==========")
        if loss_context:
            logger.info("[GRUP DISKUSI] Mode: REITERASI LOSS")

        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        api_server.start_session(session_id, signal, loss_context)

        all_messages = []

        # ── STEP 1: Yusuf analisis pembuka (kanan) ──
        logger.info("[GRUP DISKUSI] Yusuf opening...")
        yusuf_open = self._yusuf_opening(signal, market_context, loss_context)
        msg_yusuf_open = {"ai": "yusuf", "nama": "Yusuf", "pesan": yusuf_open, "side": "right"}
        all_messages.append(msg_yusuf_open)
        api_server.push_message("yusuf", "Yusuf", yusuf_open, 0, session_id)
        logger.info(f"[YUSUF OPEN] {yusuf_open}")

        # ── STEP 2: 2 ronde diskusi Arka, Nova, Zara (kiri) ──
        for ronde in range(1, 3):
            logger.info(f"[GRUP DISKUSI] --- Ronde {ronde} ---")
            new_msgs = self._run_discussion_round(
                round_num=ronde,
                chat_history=all_messages,
                signal=signal,
                market_context=market_context,
                loss_context=loss_context,
            )
            for msg in new_msgs:
                api_server.push_message(
                    ai_id=msg.get("ai", "system"),
                    nama=msg.get("nama", ""),
                    pesan=msg.get("pesan", ""),
                    ronde=ronde,
                    session_id=session_id,
                )
            all_messages.extend(new_msgs)

        # ── STEP 3: Yusuf kesimpulan akhir (kanan) ──
        logger.info("[GRUP DISKUSI] Yusuf closing...")
        yusuf_close = self._yusuf_closing(all_messages, signal)
        msg_yusuf_close = {"ai": "yusuf", "nama": "Yusuf", "pesan": yusuf_close, "side": "right"}
        all_messages.append(msg_yusuf_close)
        api_server.push_message("yusuf", "Yusuf", yusuf_close, 99, session_id)
        logger.info(f"[YUSUF CLOSE] {yusuf_close}")

        # ── Kesimpulan terstruktur (Nova) ──
        logger.info("[GRUP DISKUSI] Menyusun kesimpulan terstruktur...")
        conclusion = self._extract_panel_conclusion(all_messages, signal, market_context)
        api_server.finish_session(conclusion)

        logger.info(
            f"[GRUP DISKUSI] Selesai | consensus={conclusion.get('consensus')} | "
            f"avg_confidence={conclusion.get('avg_panel_confidence', 0):.2f}"
        )

        self._log_group_chat(all_messages, conclusion, loss_context)

        return {
            "chat_log": all_messages,
            "conclusion": conclusion,
            "avg_panel_confidence": conclusion.get("avg_panel_confidence", 0.0),
        }

    def _log_group_chat(self, messages: list, conclusion: dict, loss_context: str = ""):
        """Simpan log diskusi ke file grup_chat.log agar bisa dibaca manusia."""
        import os
        os.makedirs("logs", exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        filepath = f"logs/grup_diskusi_{timestamp}.log"
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                f.write("=" * 60 + "\n")
                f.write(f"DISKUSI PANEL AI — {timestamp}\n")
                if loss_context:
                    f.write(f"[MODE REITERASI LOSS]\n{loss_context}\n")
                f.write("=" * 60 + "\n\n")
                for msg in messages:
                    f.write(f"{msg['pesan']}\n\n")
                f.write("-" * 60 + "\n")
                f.write("KESIMPULAN:\n")
                f.write(json.dumps(conclusion, indent=2, ensure_ascii=False))
                f.write("\n")
            logger.info(f"[GRUP DISKUSI] Log disimpan: {filepath}")
        except Exception as e:
            logger.warning(f"[GRUP DISKUSI] Gagal simpan log: {e}")


    def _request_watchlist_from_panel(
        self, market_context: dict, triggered_item: dict = None, session_id: str = ""
    ) -> list:
        """
        Minta AI (Yusuf via groq_client) untuk set level watchlist berikutnya
        berdasarkan kondisi market saat ini.

        KUNCI: AI harus kasih harga PRESISI dari data (bukan rounded).
        Return: list of {"level": float, "condition": str, "reason": str, "phase": str}
        """
        import api_server

        # Konteks trigger
        trigger_context = ""
        if triggered_item:
            trigger_context = f"""
TRIGGER YANG BARU TERJADI:
- Level: {triggered_item['level']:.2f}
- Kondisi: {triggered_item['condition']}
- Phase: {triggered_item['phase']}
- Reason: {triggered_item['reason']}
- Harga saat trigger: {triggered_item.get('triggered_price', 'N/A')}
"""

        ict_data = market_context.get("ict_preliminary", {})
        current = market_context.get("current_price", {})

        # Data presisi dari ICT analyzer
        price_data = f"""
HARGA PRESISI (gunakan angka PERSIS ini, JANGAN dibulatkan):
- Bid: {current.get('bid', 0):.2f}
- Ask: {current.get('ask', 0):.2f}

STRUKTUR M15 (gunakan nilai PERSIS dari data):
- Bias: {ict_data.get('m15_bias', {}).get('direction', 'N/A')}
- Swing High: {ict_data.get('m15_bias', {}).get('last_swing_high', 0):.2f}
- Swing Low: {ict_data.get('m15_bias', {}).get('last_swing_low', 0):.2f}

ORDER BLOCKS M15:
{chr(10).join([f"  {ob['type'].upper()} OB: High={ob['high']:.2f} Low={ob['low']:.2f}" for ob in ict_data.get('m15_ob', [])[:4]])}

FVG M15:
{chr(10).join([f"  {fvg['type'].upper()} FVG: {fvg['low']:.2f}-{fvg['high']:.2f} (midpoint={fvg['midpoint']:.2f}) filled={fvg['filled']}" for fvg in ict_data.get('m15_fvg', [])[:4]])}

MSS M1: {ict_data.get('m1_mss', 'None')}
BOS M15: {ict_data.get('m15_bos', 'None')}

LIQUIDITY POOLS:
- Recent High: {ict_data.get('liquidity_pools', {}).get('recent_high', 0):.2f}
- Recent Low: {ict_data.get('liquidity_pools', {}).get('recent_low', 0):.2f}
"""

        prompt = f"""Kamu adalah Yusuf, trader ICT senior. Analisis kondisi market dan tentukan level watchlist berikutnya.
{trigger_context}
{price_data}

TUGASMU:
Berdasarkan data di atas, tentukan 2-4 level harga PRESISI yang perlu dipantau.
Level ini adalah harga dimana, jika tersentuh, analisis lebih lanjut diperlukan.

ATURAN KRITIS:
1. Gunakan angka PERSIS dari data — contoh: 71787.45, bukan 71800
2. Jika ada OB di 73585.6-73780.0 → gunakan 73585.6 atau 73780.0, BUKAN 73600
3. FVG midpoint 72320.0 → tulis 72320.0, bukan 72300
4. Setiap level harus punya alasan ICT yang jelas

KONDISI SAAT INI:
{"Belum ada BOS — set level untuk konfirmasi BOS atau reversal awal" if not ict_data.get('m15_bos') and not ict_data.get('m1_mss') else "Ada MSS/BOS — set level untuk konfirmasi entry atau invalidasi"}

WAJIB: balas HANYA dengan JSON array murni, langsung mulai dari [ tanpa penjelasan, tanpa markdown, tanpa ```json:
[
  {{
    "level": 71787.45,
    "condition": "touch|break_above|break_below",
    "reason": "alasan ICT spesifik dengan referensi ke data",
    "phase": "waiting_bos|waiting_entry|waiting_retest"
  }}
]

PENTING: gunakan angka dari data, bukan perkiraan bulat."""

        try:
            resp = self.groq_client.chat.completions.create(
                model=self.model_json,  # JSON output — pakai model_json
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,   # Low temperature = lebih presisi, tidak kreatif
                max_tokens=600,
            )
            raw = resp.choices[0].message.content.strip()
            # Parse — bisa array langsung atau wrapped dalam object
            parsed = _parse_json_safe(raw)
            if parsed is None:
                raise ValueError("Tidak bisa parse JSON watchlist")
            if isinstance(parsed, list):
                levels = parsed
            elif isinstance(parsed, dict):
                # Cari array di dalam object
                for v in parsed.values():
                    if isinstance(v, list):
                        levels = v
                        break
                else:
                    levels = []
            else:
                levels = []

            # Validasi setiap item
            valid_levels = []
            for lvl in levels:
                if isinstance(lvl, dict) and "level" in lvl and isinstance(lvl["level"], (int, float)):
                    valid_levels.append({
                        "level": float(lvl["level"]),
                        "condition": lvl.get("condition", "touch"),
                        "reason": lvl.get("reason", "")[:200],
                        "phase": lvl.get("phase", "waiting_bos"),
                    })

            logger.info(f"[WATCHLIST] AI set {len(valid_levels)} level baru")
            for lvl in valid_levels:
                logger.info(f"  → {lvl['condition'].upper()} @ {lvl['level']:.2f} [{lvl['phase']}] {lvl['reason'][:80]}")

            # Push watchlist ke API server
            api_server.update_watchlist(self.watchlist.to_api_dict())
            return valid_levels

        except Exception as e:
            err_str = str(e)
            if "429" in err_str or "rate_limit" in err_str.lower():
                logger.warning(f"[RATE LIMIT] Yusuf kena rate limit saat set watchlist | {err_str[:120]}")
            else:
                logger.error(f"[ERROR] _request_watchlist_from_panel: {err_str[:120]}")
            return []

    def run_analysis_cycle(self) -> Optional[dict]:
        """
        Satu siklus analisis + self-iteration jika ada error.
        """
        logger.info("=" * 50)
        logger.info(f"Memulai siklus analisis | {datetime.now(timezone.utc).isoformat()}")

        # 1. Ambil data market
        market_context = self.market_data.get_full_context()
        if not market_context:
            logger.error("Gagal mengambil data market")
            return None

        # 2. Analisis ICT preliminary (Python-side, sebelum AI)
        ict_pre = self.ict_analyzer.quick_check(market_context)
        market_context["ict_preliminary"] = ict_pre
        logger.info(f"ICT preliminary: {ict_pre}")

        # 3. Loop self-iteration dengan Groq AI
        signal = None
        for iteration in range(1, self.max_iterations + 1):
            try:
                signal = self._analyze_with_groq(market_context, iteration)
            except json.JSONDecodeError as e:
                error_msg = f"JSON parse error iterasi {iteration}: {e}"
                logger.error(error_msg)
                self.memory.log_iteration_error(error_msg)
                continue
            except Exception as e:
                err_str = str(e)
                if "429" in err_str or "rate_limit" in err_str.lower():
                    logger.warning(f"[RATE LIMIT] Yusuf (GROQ_API_KEY) kena rate limit di analisis ICT utama | {err_str[:150]}")
                else:
                    logger.error(f"Groq API error: {err_str[:150]}")
                self.memory.log_error(
                    error=f"Groq API error: {err_str[:100]}",
                    lesson="Periksa koneksi API dan format request",
                    context=market_context,
                )
                return None

            # Validasi sinyal
            is_valid, validation_error = self._validate_signal(signal, market_context)

            if is_valid:
                logger.info(
                    f"Sinyal valid pada iterasi {iteration}: {signal['entry_signal']}"
                )
                break
            else:
                logger.warning(
                    f"Validasi gagal (iterasi {iteration}): {validation_error}"
                )
                self.memory.log_iteration_error(validation_error)

                if iteration == self.max_iterations:
                    logger.warning("Max iterasi tercapai, skip trade")
                    self.memory.log_error(
                        error=f"Max iterasi: {validation_error}",
                        lesson=f"AI terus menghasilkan sinyal invalid: {validation_error}",
                        context={"signal": signal, "market": market_context},
                    )
                    return None

        if not signal:
            return None

        # 4. Jalankan diskusi grup AI Panel — di SEMUA kondisi
        logger.info(f"[AI PANEL] Memulai diskusi grup (sinyal: {signal.get('entry_signal', 'none')})...")
        panel_summary = self._run_ai_panel(signal, market_context)
        signal["ai_panel"] = panel_summary

        # Log consensus panel ke trading log
        consensus = panel_summary.get("conclusion", {}).get("consensus", "N/A")
        avg_conf = panel_summary.get("avg_panel_confidence", 0.0)
        logger.info(f"[AI PANEL] Consensus: {consensus} | Avg Confidence: {avg_conf:.2f}")

        # 5. Eksekusi trade jika ada sinyal
        if signal["entry_signal"] in ["buy", "sell"]:
            # Sync balance live dari Bybit sebelum kalkulasi
            live_balance = self.executor.get_account_balance()
            if live_balance > 0:
                self.risk_manager.account_balance = live_balance
            lot_size = self.risk_manager.calculate_lot_size(
                entry=signal["entry_price"],
                stop_loss=signal["stop_loss"],
                symbol=self.symbol,
            )

            trade_result = self.executor.execute(
                direction=signal["entry_signal"],
                entry_price=signal["entry_price"],
                stop_loss=signal["stop_loss"],
                take_profit=signal["take_profit"],
                lot_size=lot_size,
                signal=signal,
            )

            # Simpan ke memory
            self.memory.log_trade(
                symbol=self.symbol,
                direction=signal["entry_signal"],
                setup=signal.get("bias_reason", ""),
                entry=signal["entry_price"],
                sl=signal["stop_loss"],
                tp=signal["take_profit"],
                rr=signal["risk_reward"],
                confidence=signal["confidence"],
                notes=signal.get("entry_reason", ""),
                trade_id=trade_result.get("trade_id"),
            )

            logger.info(
                f"Trade dieksekusi: {signal['entry_signal'].upper()} "
                f"@ {signal['entry_price']} | SL: {signal['stop_loss']} | TP: {signal['take_profit']}"
            )

        else:
            logger.info(f"No trade: {signal.get('skip_reason', 'Tidak ada setup')}")

        return signal

    def monitor_open_trades(self):
        """Monitor trade yang aktif dan update hasil ke memory."""
        closed = self.executor.check_closed_trades()
        for trade in closed:
            self.memory.update_trade_result(
                trade_id=trade["trade_id"],
                result=trade["result"],  # "win" | "loss"
                pnl=trade["pnl"],
                exit_price=trade["exit_price"],
                exit_reason=trade["exit_reason"],
            )
            logger.info(
                f"Trade closed: {trade['result'].upper()} | PnL: {trade['pnl']:.2f}"
            )

            # Jika loss, minta AI untuk introspeksi
            if trade["result"] == "loss":
                # Loss debrief — 4 AI evaluasi apa yang salah
                try:
                    import api_server
                    debrief = loss_debrief(
                        clients=self.ai_panel + [self.groq_client],
                        models=self.model_panel + [self.model_json],
                        closed_trade=trade,
                        trade_context={}
                    )
                    self.memory.log_error(
                        error=debrief.get("root_cause", "unknown"),
                        lesson=debrief.get("lesson", ""),
                        context={"culprit": debrief.get("culprit"), "new_rule": debrief.get("new_rule")},
                    )
                    # Push debrief ke grup chat
                    session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S") + "_debrief"
                    api_server.start_session(session_id, trade, "LOSS DEBRIEF")
                    for msg in debrief.get("chat_log", []):
                        api_server.push_message(
                            msg.get("ai","system"), msg.get("nama",""), msg.get("pesan",""),
                            msg.get("ronde",1), session_id
                        )
                    api_server.finish_session({
                        "consensus": "loss_debrief",
                        "poin_debat": debrief.get("culprit",""),
                        "risiko_utama": debrief.get("root_cause",""),
                        "kondisi_reentry": debrief.get("new_rule",""),
                    })
                    logger.info(f"[DEBRIEF] Culprit: {debrief.get('culprit')} | {debrief.get('lesson')}")
                except Exception as e:
                    logger.error(f"[DEBRIEF ERROR] {e}")
                    self._post_trade_analysis(trade)

    def _post_trade_analysis(self, closed_trade: dict):
        """Minta Groq AI untuk menganalisis trade yang loss."""
        logger.info("Melakukan post-trade analysis untuk trade yang loss...")
        try:
            response = self.groq_client.chat.completions.create(
                model=self.model_json,  # JSON output — pakai model_json
                messages=[
                    {
                        "role": "system",
                        "content": "Kamu adalah mentor trading ICT yang menganalisis kesalahan untuk pembelajaran.",
                    },
                    {
                        "role": "user",
                        "content": f"""
Analisis trade yang loss berikut dan identifikasi kesalahan spesifik:

Trade Data:
{json.dumps(closed_trade, indent=2)}

WAJIB: balas HANYA JSON murni, langsung dari {{ tanpa penjelasan atau markdown:
{{
  "kesalahan_utama": "deskripsi kesalahan",
  "pelajaran": "apa yang harus diperbaiki",
  "kondisi_yang_seharusnya_skip": "kapan seharusnya tidak entry",
  "perbaikan_untuk_besok": "rule tambahan yang harus diterapkan"
}}
""",
                    },
                ],
                temperature=0.3,
                max_tokens=512,
            )

            analysis = _parse_json_safe(response.choices[0].message.content)
            if analysis is None:
                raise ValueError("Tidak bisa parse JSON post-trade")
            self.memory.log_error(
                error=analysis["kesalahan_utama"],
                lesson=analysis["pelajaran"],
                context={
                    "trade": closed_trade,
                    "skip_condition": analysis["kondisi_yang_seharusnya_skip"],
                    "improvement": analysis["perbaikan_untuk_besok"],
                },
            )
            logger.info(f"Post-trade lesson saved: {analysis['pelajaran']}")

            # Trigger AI Panel reiterasi — 3 analis kembangkan insight dari loss ini
            logger.info("[AI PANEL] Memulai reiterasi dari 3 analis setelah loss...")
            loss_context = (
                f"Trade loss terdeteksi.\n"
                f"Kesalahan utama: {analysis['kesalahan_utama']}\n"
                f"Pelajaran: {analysis['pelajaran']}\n"
                f"Kondisi yang seharusnya skip: {analysis['kondisi_yang_seharusnya_skip']}\n"
                f"Perbaikan untuk ke depan: {analysis['perbaikan_untuk_besok']}"
            )
            # Buat sinyal dummy dari closed_trade agar panel bisa bekerja
            dummy_signal = {
                "entry_signal": closed_trade.get("direction", "none"),
                "entry_price": closed_trade.get("entry", 0),
                "stop_loss": closed_trade.get("sl", 0),
                "take_profit": closed_trade.get("tp", 0),
                "bias_reason": closed_trade.get("setup", ""),
                "entry_reason": closed_trade.get("notes", ""),
            }
            dummy_context = {"symbol": self.symbol, "current_price": {}}
            panel_reiter = self._run_ai_panel(dummy_signal, dummy_context, loss_context)
            self.memory.log_error(
                error=f"[PANEL REITERASI] Post-loss panel selesai",
                lesson=f"avg_panel_confidence={panel_reiter['avg_panel_confidence']:.2f}",
                context={"panel": panel_reiter["panel_results"]},
            )
            logger.info(
                f"[AI PANEL REITERASI] Selesai | avg_confidence={panel_reiter['avg_panel_confidence']:.2f}"
            )

        except Exception as e:
            logger.error(f"Post-trade analysis failed: {e}")

    def _run_specialist_cycle(self, market_context: dict, triggered_item: dict = None) -> dict | None:
        """
        Jalankan 4 AI spesialis sesuai fase saat ini.
        Setiap fase hanya memanggil 1 AI — hemat token.

        Fase:
          m15_scan    → AI-1 analisis M15, set watchlist M15
          idm_hunt    → AI-2 cari IDM, set notif level IDM
          bos_guard   → AI-3 konfirmasi BOS/MSS setelah IDM disentuh
          entry_sniper → AI-4 tentukan entry, SL, TP
        """
        import api_server

        ict_data = market_context.get("ict_preliminary", {})
        current_price = market_context.get("current_price", {}).get("bid", 0)
        m1_candles = market_context.get("m1", {}).get("candles", [])
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

        phase = self._phase
        logger.info(f"[SPECIALIST] Fase: {phase.upper()} | Harga: {current_price:.2f}")

        result = {"phase": phase, "session_id": session_id}

        # ── FASE 1: M15 SCAN (AI-1) ──────────────────────
        if phase == "m15_scan":
            out = ai1_m15_analysis(
                self.ai_panel[0], self.model_ai1,
                ict_data, current_price
            )
            self._current_bias = out.get("bias", "neutral")
            result["ai1"] = out

            # Push ke grup chat
            if out.get("chat_msg"):
                api_server.start_session(session_id, {}, "")
                api_server.push_message("ai1", "🔭 AI-1 M15", out["chat_msg"], 1, session_id)
                api_server.finish_session({"consensus": f"bias_{self._current_bias}"})

            # Set watchlist dari AI-1
            wl = out.get("watchlist", [])
            if wl:
                self.watchlist.clear_untriggered()
                self.watchlist.add_many(wl, session_id)
                self._phase = "idm_hunt"
                logger.info(f"[SPECIALIST] AI-1 selesai | Bias: {self._current_bias} | "
                           f"Watchlist: {len(wl)} level | Fase → idm_hunt")
            else:
                logger.warning("[SPECIALIST] AI-1 tidak set watchlist — retry M15 scan")

        # ── FASE 2: IDM HUNT (AI-2) ──────────────────────
        elif phase == "idm_hunt":
            # Cari IDM via Python analyzer dulu (lebih presisi, tanpa token)
            idm_py = self.ict_analyzer.get_latest_idm(m1_candles, self._current_bias)

            out = ai2_idm_hunter(
                self.ai_panel[1], self.model_ai2,
                ict_data, idm_py, self._current_bias, current_price
            )
            self._current_idm_level = out.get("idm_level", 0)
            result["ai2"] = out

            # Push ke grup chat
            if out.get("chat_msg"):
                api_server.start_session(session_id, {}, "")
                api_server.push_message("ai2", "🔍 AI-2 IDM", out["chat_msg"], 1, session_id)
                api_server.finish_session({"consensus": "idm_found" if out.get("idm_valid") else "idm_hunting"})

            wl = out.get("watchlist", [])
            if wl and self._current_idm_level > 0:
                self.watchlist.add_many(wl, session_id)
                self._phase = "bos_guard"
                logger.info(f"[SPECIALIST] AI-2 selesai | IDM level: {self._current_idm_level:.2f} | "
                           f"Fase → bos_guard")
            else:
                logger.warning("[SPECIALIST] AI-2 IDM belum ditemukan — tetap di fase idm_hunt")

        # ── FASE 3: BOS/MSS GUARD (AI-3) ─────────────────
        elif phase == "bos_guard":
            # Cek BOS M1 via Python analyzer
            bos_py = self.ict_analyzer.check_bos_m1(
                m1_candles, self._current_bias, self._current_idm_level
            )

            out = ai3_bos_mss_guard(
                self.ai_panel[2], self.model_ai3,
                ict_data, bos_py, self._current_idm_level,
                self._current_bias, current_price
            )
            decision = out.get("decision", "wait")
            result["ai3"] = out

            # Push ke grup
            if out.get("chat_msg"):
                api_server.start_session(session_id, {}, "")
                api_server.push_message("ai3", "⚡ AI-3 BOS/MSS", out["chat_msg"], 1, session_id)
                api_server.finish_session({"consensus": decision})

            if decision == "bos":
                self._current_bos_level = bos_py["level"] if bos_py else current_price
                self._phase = "entry_sniper"
                logger.info(f"[SPECIALIST] AI-3: BOS dikonfirmasi @ {self._current_bos_level:.2f} | "
                           f"Fase → entry_sniper")
            elif decision == "mss":
                # Reset ke IDM hunt — cari IDM baru
                self._current_idm_level = 0
                self._phase = "idm_hunt"
                logger.info("[SPECIALIST] AI-3: MSS — kembali ke IDM hunt")
                # Set watchlist dari AI-3
                wl = out.get("watchlist", [])
                if wl:
                    self.watchlist.add_many(wl, session_id)
            else:
                # Wait — set watchlist konfirmasi
                wl = out.get("watchlist", [])
                if wl:
                    self.watchlist.add_many(wl, session_id)
                logger.info("[SPECIALIST] AI-3: Menunggu konfirmasi BOS/MSS")

        # ── FASE 4: ENTRY SNIPER (AI-4) ──────────────────
        elif phase == "entry_sniper":
            # MSNR dari Python analyzer
            msnr_dir = "support" if self._current_bias == "bullish" else "resistance"
            msnr_levels = self.ict_analyzer.msnr_level(m1_candles, msnr_dir)

            # Trade memory untuk referensi AI-4
            trade_mem = self.memory.get_recent_trades(limit=5)

            out = ai4_entry_sniper(
                self.groq_client, self.model_json,
                ict_data, msnr_levels, current_price,
                self._current_bias, self._current_bos_level,
                trade_mem
            )
            result["ai4"] = out

            # Push ke grup
            if out.get("chat_msg"):
                api_server.start_session(session_id, {}, "")
                api_server.push_message("yusuf", "🎯 AI-4 Entry", out["chat_msg"], 1, session_id)
                api_server.finish_session({
                    "consensus": "setuju_lanjut" if out.get("confidence", 0) >= 0.65 else "hati_hati",
                    "entry_ideal_zona": str(out.get("entry", "")),
                    "entry_ideal_timing": out.get("setup_type", ""),
                    "risiko_utama": f"SL di {out.get('sl',0):.2f} ({out.get('sl_reason','')})",
                    "avg_panel_confidence": out.get("confidence", 0),
                })

            # Eksekusi kalau confidence cukup
            entry = out.get("entry", 0)
            sl    = out.get("sl", 0)
            tp    = out.get("tp", 0)
            conf  = out.get("confidence", 0)

            if entry > 0 and sl > 0 and tp > 0 and conf >= 0.6:
                logger.info(f"[SPECIALIST] AI-4 Entry signal | {self._current_bias.upper()} "
                           f"@ {entry:.2f} | SL {sl:.2f} | TP {tp:.2f} | conf {conf:.0%}")

                # Sync balance
                live_balance = self.executor.get_account_balance()
                if live_balance > 0:
                    self.risk_manager.account_balance = live_balance

                lot_size = self.risk_manager.calculate_lot_size(
                    entry=entry, stop_loss=sl, symbol=self.symbol
                )
                direction = "buy" if self._current_bias == "bullish" else "sell"

                trade_result = self.executor.execute(
                    direction=direction,
                    entry_price=entry,
                    stop_loss=sl,
                    take_profit=tp,
                    lot_size=lot_size,
                    signal={"entry_signal": direction, "confidence": conf,
                            "bias_reason": f"AI-4 {out.get('setup_type','')}"},
                )
                self.memory.log_trade(
                    symbol=self.symbol, direction=direction,
                    setup=out.get("setup_type", ""), entry=entry,
                    sl=sl, tp=tp, rr=out.get("rr", 0),
                    confidence=conf, notes=out.get("chat_msg", ""),
                    trade_id=trade_result.get("trade_id"),
                )
                result["executed"] = True
            else:
                logger.info(f"[SPECIALIST] AI-4 skip eksekusi | conf={conf:.0%} | entry={entry} sl={sl} tp={tp}")
                result["executed"] = False

            # Reset ke M15 scan untuk siklus berikutnya
            self._phase = "m15_scan"
            self._current_idm_level = 0
            self._current_bos_level = 0
            logger.info("[SPECIALIST] Siklus selesai — reset ke m15_scan")

        api_server.update_watchlist(self.watchlist.to_api_dict())
        return result

    async def run(self):
        """
        Main loop — trigger-based, bukan time-based.

        Alur:
        1. Siklus pertama: analisis awal + AI set watchlist level
        2. Setiap siklus berikutnya: cek harga vs watchlist
           - Tidak ada trigger → log harga saja, TIDAK panggil AI
           - Ada trigger → panggil AI panel diskusi → AI set watchlist baru
        3. AI hanya dipanggil saat ada EVENT, bukan setiap menit
        """
        import api_server

        logger.info("=" * 60)
        logger.info("ICT Trading Bot — Trigger-Based System")
        logger.info(f"Symbol: {self.symbol} | Mode: {'PAPER' if self.paper_trading else 'LIVE'}")
        logger.info("=" * 60)

        while True:
            try:
                # Ambil data market
                market_context = self.market_data.get_full_context()
                if not market_context:
                    logger.error("Gagal ambil data market")
                    await asyncio.sleep(self.scan_interval)
                    continue

                current_price = market_context.get("current_price", {}).get("bid", 0)
                if not current_price:
                    await asyncio.sleep(self.scan_interval)
                    continue

                # ICT preliminary check (Python-side, tidak pakai token)
                ict_pre = self.ict_analyzer.quick_check(market_context)
                market_context["ict_preliminary"] = ict_pre

                # ── SCAN HARGA & TRIGGER ──────────────────────────
                triggered = self.watchlist.check(current_price, self._prev_price)

                if triggered:
                    for item in triggered:
                        logger.info(
                            f"[TRIGGER] 🔔 {item['condition'].upper()} @ {item['level']:.2f} | "
                            f"for={item.get('for_ai','?')} | {item['reason']}"
                        )
                    logger.info(f"[SPECIALIST] {len(triggered)} trigger — jalankan fase {self._phase}")

                    try:
                        self._run_specialist_cycle(market_context, triggered[-1])
                        self._initial_analysis_done = True
                    except Exception as e:
                        err = str(e)
                        if "429" in err or "rate_limit" in err.lower() or "413" in err:
                            logger.warning(f"[RATE LIMIT] Specialist cycle kena limit: {err[:100]}")
                        else:
                            logger.error(f"[ERROR] Specialist cycle: {err[:150]}")

                elif not self._initial_analysis_done:
                    # Startup — langsung jalankan AI-1 M15 scan
                    logger.info(f"[SPECIALIST] Startup — jalankan AI-1 M15 scan")
                    try:
                        self._run_specialist_cycle(market_context)
                        self._initial_analysis_done = True
                    except Exception as e:
                        err = str(e)
                        if "429" in err or "413" in err or "rate_limit" in err.lower():
                            logger.warning(f"[RATE LIMIT] Startup scan: {err[:100]}")
                        else:
                            logger.error(f"[ERROR] Startup scan: {err[:150]}")
                else:
                    # Tidak ada trigger — log harga saja, 0 token
                    active = self.watchlist.get_active()
                    logger.info(
                        f"[BOT] 💰 {current_price:.2f} | "
                        f"Fase: {self._phase} | "
                        f"Watchlist: {len(active)} level aktif"
                    )
                    if active:
                        nearest = min(active, key=lambda x: abs(x["level"] - current_price))
                        dist = abs(nearest["level"] - current_price)
                        logger.info(
                            f"[BOT] Terdekat: {nearest['level']:.2f} "
                            f"({nearest['condition']}, for {nearest.get('for_ai','?')}) | "
                            f"Jarak: {dist:.2f}"
                        )

                # Monitor trade aktif
                self.monitor_open_trades()

                # Update harga sebelumnya
                self._prev_price = current_price

                # Stats
                stats = self.memory.get_stats()
                logger.info(
                    f"Stats | Total: {stats['total_trades']} | "
                    f"Win: {stats['wins']} | Loss: {stats['losses']} | "
                    f"WR: {stats['win_rate']:.0%}"
                )

            except KeyboardInterrupt:
                logger.info("Bot dihentikan oleh user")
                break
            except Exception as e:
                logger.error(f"Unexpected error: {e}", exc_info=True)
                self.memory.log_error(
                    error=str(e),
                    lesson="Unhandled exception",
                    context={"type": type(e).__name__},
                )

            logger.info(f"Menunggu {self.scan_interval} detik...\n")
            await asyncio.sleep(self.scan_interval)
