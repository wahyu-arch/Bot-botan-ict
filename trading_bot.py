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

        self.symbol = os.getenv("TRADING_SYMBOL", "XAUUSD")
        self.paper_trading = os.getenv("PAPER_TRADING", "true").lower() == "true"
        self.max_iterations = int(os.getenv("MAX_AI_ITERATIONS", "3"))
        self.scan_interval = int(os.getenv("SCAN_INTERVAL_SECONDS", "60"))

        self.market_data = MarketDataFetcher(self.symbol)
        self.ict_analyzer = ICTAnalyzer()
        self.memory = MemorySystem()
        self.risk_manager = RiskManager()
        self.executor = TradeExecutor(paper_mode=self.paper_trading)

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

RESPONSE FORMAT (JSON ketat, tanpa teks lain):
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
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": self._build_system_prompt()},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.1,  # Low temperature untuk konsistensi
            max_tokens=1024,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content
        logger.info(f"Groq response: {raw}")
        return json.loads(raw)

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
        Jalankan 1 ronde diskusi: setiap AI merespons chat history yang ada.
        Urutan bicara: Arka → Nova → Zara (tiap AI lihat pesan sebelumnya di ronde ini juga).
        Returns list pesan baru di ronde ini.
        """
        import concurrent.futures

        round_messages = []
        ai_order = ["AI-1", "AI-2", "AI-3"]

        # Di ronde 1: paralel (belum ada konteks satu sama lain di ronde ini)
        # Di ronde 2-3: sekuensial agar tiap AI bisa baca pesan sebelumnya di ronde tsb
        if round_num == 1:
            def call_opening(ai_id):
                p = self.AI_PERSONAS[ai_id]
                client = self.ai_panel[int(ai_id[-1]) - 1]
                sys_prompt = self._build_persona_system_prompt(ai_id, signal, market_context, loss_context)

                history_text = self._format_history(chat_history)
                user_msg = f"""=== RIWAYAT DISKUSI SEJAUH INI ===
{history_text if history_text else "(Belum ada diskusi — kamu yang mulai duluan sebagai pembuka)"}

=== GILIRANMU ===
Ini Ronde {round_num}. Berikan opening statement kamu tentang setup ini.
Mulai dengan nama kamu, lalu langsung ke analisis. Contoh format:
"{p['singkatan']}: [isi pesanmu]"

Tulis HANYA pesanmu saja (plain text, bukan JSON)."""

                resp = client.chat.completions.create(
                    model="llama-3.3-70b-versatile",
                    messages=[
                        {"role": "system", "content": sys_prompt},
                        {"role": "user", "content": user_msg},
                    ],
                    temperature=0.7,
                    max_tokens=300,
                )
                return ai_id, resp.choices[0].message.content.strip()

            with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
                futures = {executor.submit(call_opening, ai_id): ai_id for ai_id in ai_order}
                raw_results = {}
                for future in concurrent.futures.as_completed(futures):
                    ai_id, msg = future.result()
                    raw_results[ai_id] = msg
            # Urutkan sesuai ai_order
            for ai_id in ai_order:
                msg = raw_results.get(ai_id, "")
                round_messages.append({"ai": ai_id, "nama": self.AI_PERSONAS[ai_id]["nama"], "pesan": msg})
                logger.info(f"[GRUP RONDE {round_num}] {msg}")

        else:
            # Ronde 2 & 3: sekuensial — tiap AI lihat pesan sebelumnya di ronde ini
            for ai_id in ai_order:
                p = self.AI_PERSONAS[ai_id]
                client = self.ai_panel[int(ai_id[-1]) - 1]
                sys_prompt = self._build_persona_system_prompt(ai_id, signal, market_context, loss_context)

                # Gabung: semua history + pesan ronde ini yang sudah terbit
                full_history = chat_history + round_messages
                history_text = self._format_history(full_history)

                if round_num == 3:
                    task_note = (
                        "Ini RONDE TERAKHIR. Berikan kesimpulan akhirmu yang mencakup: "
                        "(1) setuju/tidak dengan sinyal utama, "
                        "(2) zona/harga entry ideal menurutmu dan kenapa, "
                        "(3) timing entry terbaik, "
                        "(4) satu risiko terbesar yang harus diwaspadai, "
                        "(5) kondisi reentry jika loss. "
                        "Tetap gaya WA, padat dan to-the-point."
                    )
                else:
                    task_note = (
                        f"Ini Ronde {round_num}. Respons pendapat yang sudah disampaikan — "
                        "bisa setuju, bantah, atau tanya ke salah satu analis lain. "
                        "Fokus ke: zona entry ideal, timing, dan risiko tersembunyi."
                    )

                user_msg = f"""=== RIWAYAT DISKUSI ===
{history_text}

=== GILIRANMU ===
{task_note}

Mulai dengan namamu. Format:
"{p['singkatan']}: [isi pesanmu]"

Tulis HANYA pesanmu saja (plain text)."""

                try:
                    resp = client.chat.completions.create(
                        model="llama-3.3-70b-versatile",
                        messages=[
                            {"role": "system", "content": sys_prompt},
                            {"role": "user", "content": user_msg},
                        ],
                        temperature=0.65,
                        max_tokens=350,
                    )
                    msg = resp.choices[0].message.content.strip()
                except Exception as e:
                    msg = f"{p['singkatan']}: [error: {e}]"

                round_messages.append({"ai": ai_id, "nama": p["nama"], "pesan": msg})
                logger.info(f"[GRUP RONDE {round_num}] {msg}")

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

Berikan ringkasan dalam format JSON (jangan ubah entry_price/SL/TP):
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
                model="llama-3.3-70b-versatile",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.2,
                max_tokens=600,
                response_format={"type": "json_object"},
            )
            return json.loads(resp.choices[0].message.content)
        except Exception as e:
            logger.warning(f"[PANEL CONCLUSION] error: {e}")
            return {"consensus": "error", "error": str(e), "avg_panel_confidence": 0.0}

    def _run_ai_panel(self, signal: dict, market_context: dict, loss_context: str = "") -> dict:
        """
        Jalankan diskusi grup WhatsApp 3 ronde antara Arka, Nova, Zara.
        Returns: semua pesan + kesimpulan terstruktur.
        """
        import api_server
        from datetime import timezone

        logger.info("[GRUP DISKUSI] ========== Mulai diskusi 3 ronde ==========")
        if loss_context:
            logger.info(f"[GRUP DISKUSI] Mode: REITERASI LOSS")

        # Buat session ID unik
        session_id = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        api_server.start_session(session_id, signal, loss_context)

        all_messages = []

        for ronde in range(1, 4):
            logger.info(f"[GRUP DISKUSI] --- Ronde {ronde} ---")
            new_msgs = self._run_discussion_round(
                round_num=ronde,
                chat_history=all_messages,
                signal=signal,
                market_context=market_context,
                loss_context=loss_context,
            )
            # Push tiap pesan ke API server secara realtime
            for msg in new_msgs:
                api_server.push_message(
                    ai_id=msg.get("ai", "system"),
                    nama=msg.get("nama", ""),
                    pesan=msg.get("pesan", ""),
                    ronde=ronde,
                    session_id=session_id,
                )
            all_messages.extend(new_msgs)

        logger.info("[GRUP DISKUSI] --- Menyusun kesimpulan ---")
        conclusion = self._extract_panel_conclusion(all_messages, signal, market_context)
        api_server.finish_session(conclusion)

        logger.info(
            f"[GRUP DISKUSI] Selesai | consensus={conclusion.get('consensus')} | "
            f"avg_confidence={conclusion.get('avg_panel_confidence', 0):.2f}"
        )

        # Log semua chat ke file terpisah agar mudah dibaca
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
                error_msg = f"Groq API error: {e}"
                logger.error(error_msg)
                self.memory.log_error(
                    error=error_msg,
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
                self._post_trade_analysis(trade)

    def _post_trade_analysis(self, closed_trade: dict):
        """Minta Groq AI untuk menganalisis trade yang loss."""
        logger.info("Melakukan post-trade analysis untuk trade yang loss...")
        try:
            response = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
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

Berikan analisis dalam format JSON:
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
                response_format={"type": "json_object"},
            )

            analysis = json.loads(response.choices[0].message.content)
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

    async def run(self):
        """Main loop bot trading."""
        logger.info("=" * 60)
        logger.info("ICT Trading Bot dengan Groq AI + Memory System")
        logger.info(f"Symbol: {self.symbol} | Mode: {'PAPER' if self.paper_trading else 'LIVE'}")
        logger.info("=" * 60)

        while True:
            try:
                # Monitor trade aktif
                self.monitor_open_trades()

                # Jalankan satu siklus analisis
                self.run_analysis_cycle()

                # Tampilkan statistik memori
                stats = self.memory.get_stats()
                logger.info(
                    f"Stats | Total: {stats['total_trades']} | "
                    f"Win: {stats['wins']} | Loss: {stats['losses']} | "
                    f"WR: {stats['win_rate']:.0%} | Errors logged: {stats['total_errors']}"
                )

            except KeyboardInterrupt:
                logger.info("Bot dihentikan oleh user")
                break
            except Exception as e:
                logger.error(f"Unexpected error di main loop: {e}", exc_info=True)
                self.memory.log_error(
                    error=str(e),
                    lesson="Unhandled exception - perlu review kode",
                    context={"type": type(e).__name__},
                )

            logger.info(f"Menunggu {self.scan_interval} detik...\n")
            await asyncio.sleep(self.scan_interval)
