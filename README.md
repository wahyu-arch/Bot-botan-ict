# 🤖 ICT Trading Bot — Groq AI + Bybit + Memory Self-Iteration

Bot trading otomatis untuk Bybit Linear Perpetual menggunakan:
- **Groq AI (LLaMA 3.3 70B)** — otak analisis & keputusan
- **Strategi ICT**: M15 arah market, M1 konfirmasi entry
- **Risk 1% per trade** — fixed, dikalkulasi otomatis dari balance
- **Memory Self-Iteration** — AI belajar dari kesalahan sendiri
- **Railway-ready** — deploy satu klik

---

## 🧠 Alur Self-Iteration

```
M15 + M1 Candles (Bybit Klines)
        ↓
ICT Analyzer  →  OB, FVG, BOS, MSS, Liquidity
        ↓
Groq AI  ←  memory errors dari trade sebelumnya
        ↓
Validasi: SL/TP valid? RR ≥ 1.5? Confidence ≥ 60%?
     ↙ invalid            ↘ valid
Log error          Kalkulasi qty (1% risk)
Iterasi ulang            ↓
(max 3x)          Kirim ke Bybit API
                         ↓
                  Monitor → SL/TP hit
                         ↓
                  Jika LOSS → AI introspeksi
                         ↓
                  Simpan lesson → dipakai besok
```

---

## 📊 Strategi ICT

| Konsep | TF | Fungsi |
|--------|-----|--------|
| Market Structure HH/HL/LH/LL | M15 | Bias bullish/bearish |
| Order Block (OB) | M15 | Area supply/demand |
| Fair Value Gap (FVG) | M15+M1 | Imbalance, magnet harga |
| Break of Structure (BOS) | M15 | Konfirmasi trend |
| Market Structure Shift (MSS) | M1 | Trigger entry |
| Liquidity Pools | M15 | Target TP |

**Entry hanya jika minimum 2 konfluensi ICT terpenuhi.**

---

## 💰 Risk Management (1% Fixed)

```
Qty = (Balance × 1%) / SL_distance_USD

Contoh BTCUSDT:
  Balance    = $10,000
  Risk 1%    = $100
  Entry      = $65,000
  SL         = $64,350  → SL distance = $650
  Qty        = $100 / $650 = 0.154 BTC
  Max loss   = $100 (persis 1% balance)
```

Balance di-sync live dari Bybit sebelum tiap kalkulasi.

---

## 🚀 Deploy ke Railway

### 1. Siapkan Bybit API Key

**Testnet (recommended untuk mulai):**
- Daftar di [testnet.bybit.com](https://testnet.bybit.com)
- API → Create Key → centang "Contract" (read + write)

**Mainnet:**
- [bybit.com](https://www.bybit.com) → API Management
- Izinkan: Contract Orders (bukan withdrawal!)
- Whitelist IP Railway jika perlu

### 2. Siapkan Groq API Key
- Daftar gratis di [console.groq.com](https://console.groq.com)
- Create API Key

### 3. Deploy

```bash
# Push ke GitHub dulu
git init && git add . && git commit -m "init"
git push origin main

# Lalu hubungkan repo di railway.app
# Atau via CLI:
npm i -g @railway/cli
railway login
railway init
railway up
```

### 4. Set Environment Variables di Railway

```
GROQ_API_KEY        = gsk_xxxxxxxxxxxx
BYBIT_API_KEY       = xxxxxxxxxxxxxxxxxx
BYBIT_API_SECRET    = xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx
BYBIT_TESTNET       = true
TRADING_SYMBOL      = BTCUSDT
PAPER_TRADING       = true
DATA_SOURCE         = bybit
ACCOUNT_BALANCE     = 10000
MAX_QTY             = 0.5
SCAN_INTERVAL_SECONDS = 60
MAX_AI_ITERATIONS   = 3
```

---

## 📁 Struktur Project

```
trading-bot/
├── main.py                 ← Entry point
├── railway.toml            ← Config Railway
├── nixpacks.toml           ← Build Python 3.11
├── requirements.txt        ← groq, pybit, requests
├── .env.example            ← Template env vars
├── src/
│   ├── trading_bot.py      ← Orchestrator utama
│   ├── ict_analyzer.py     ← OB, FVG, BOS, MSS detection
│   ├── memory_system.py    ← Persistent memory + lessons
│   ├── market_data.py      ← Bybit klines fetcher
│   ├── risk_manager.py     ← Qty calc (1% risk fixed)
│   └── trade_executor.py   ← Bybit order + paper mode
├── data/
│   ├── trade_memory.json   ← Memory persisten
│   └── paper_trades.json   ← Riwayat paper trade
└── logs/
    └── trading_bot.log
```

---

## 📋 Contoh Log Normal

```
09:30:01 | INFO | ══════════════════════════════════════
09:30:01 | INFO | ICT Trading Bot dengan Groq AI + Memory System
09:30:01 | INFO | Symbol: BTCUSDT | Mode: PAPER
09:30:02 | INFO | Memulai siklus analisis
09:30:03 | INFO | ICT preliminary: bias=bearish | OBs=2 | FVGs=3
09:30:04 | INFO | Mengirim ke Groq AI (iterasi 1)...
09:30:05 | INFO | Sinyal valid iterasi 1: sell
09:30:05 | INFO | Risk Calc | Balance: $10000 | Risk: 1% = $100 | SL dist: $420 | Qty: 0.238 BTC
09:30:05 | INFO | PAPER #A3F1: SELL 0.238 @ 65200 | SL: 65620 | TP: 63960 | Max loss: $99.96
09:30:06 | INFO | Stats | Total: 15 | Win: 9 | Loss: 4 | WR: 60% | Errors: 6
```

### Contoh Self-Iteration:
```
09:45:04 | WARNING | Validasi gagal (iterasi 1): RR terlalu rendah: 1.2 (min 1.5:1)
09:45:05 | INFO    | Mengirim ke Groq AI (iterasi 2)...
09:45:06 | INFO    | Sinyal valid iterasi 2: sell
```

### Contoh Post-Trade Learning:
```
10:12:33 | INFO | Trade closed: LOSS | PnL: -$99.80
10:12:34 | INFO | Post-trade lesson: "Entry terlalu terburu-buru sebelum MSS M1 konfirmasi"
10:12:34 | INFO | Lesson disimpan ke memory — akan digunakan di siklus berikutnya
```

---

## ⚠️ Disclaimer

Selalu mulai dengan `PAPER_TRADING=true` dan `BYBIT_TESTNET=true`.
Bot ini untuk edukasi. Trading crypto memiliki risiko tinggi kehilangan modal.
