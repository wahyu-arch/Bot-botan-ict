# 🤖 Bot Botan ICT — Multi-AI Trading System

Bot trading otomatis Bybit Futures menggunakan chain analisis 5 AI dengan arsitektur **event-driven + persistent state**.

---

## 🧠 Arsitektur AI

```
Hiura (H1 analyst)
  ↓ force_phase + notify
Senanan (IDM M5 hunter)
  ↓ force_phase + notify
Shina (BOS/MSS M5 guard)
  ↓ force_phase + notify
Yusuf (Entry sniper)
  ↓ place_order
Katyusha (Supervisor — opsional, bisa dimatikan)
```

Setiap AI punya **authority** berbeda:
- **Hiura** → force fase, tambah watchlist, notify Senanan
- **Senanan** → force fase, tambah watchlist, notify Shina
- **Shina** → force semua fase, notify Yusuf / reset ke Senanan
- **Yusuf** → eksekusi order, reset ke h1_scan setelah entry
- **Katyusha** → authority penuh, bisa on/off via tombol di UI

Chain berjalan **mandiri tanpa Katyusha** — bisa dimatikan di panel chat.

---

## 📊 Fase Trading

```
h1_scan → fvg_wait → idm_hunt → bos_guard → entry_sniper
```

| Fase | AI | Trigger |
|---|---|---|
| `h1_scan` | Hiura | Setiap M5 close |
| `fvg_wait` | Hiura | BOS H1 ditemukan |
| `idm_hunt` | Senanan | Harga masuk zona FVG |
| `bos_guard` | Shina | IDM M5 disentuh |
| `entry_sniper` | Yusuf | MSS M5 confirmed |

---

## ⚙️ Deploy ke Railway / Render / Fly.io

```bash
# Clone repo
git clone <repo_url>
cd bot-botan-ict

# Set environment variables (lihat .env.example)
# Deploy
railway up
```

### Pindah dari Railway ke platform lain:
1. Copy semua env variables dari Railway Dashboard → Variables
2. Pastikan `data/` folder di-mount sebagai persistent volume
   - Railway: sudah otomatis via volume mount
   - Render: tambah Disk di Settings → mount path `/app/data`
   - Fly.io: `fly volumes create data --size 1`
3. Port yang diexpose: `8080` (Flask API + HTML dashboard)

---

## 🔑 Environment Variables

Lihat `.env.example` untuk daftar lengkap. Yang wajib:

| Variable | Keterangan |
|---|---|
| `GROQ_API_KEY` | API key Groq (Hiura, Senanan, Shina, Yusuf) |
| `BYBIT_API_KEY` | Bybit API key |
| `BYBIT_API_SECRET` | Bybit API secret |
| `TRADING_SYMBOL` | Symbol trading (contoh: `BTCUSDT`) |
| `PAPER_TRADING` | `true` = paper, `false` = live |
| `OPENROUTER_API_KEY` | Opsional — untuk Katyusha (Claude Sonnet) |

---

## 📁 File JSON Penting

Semua di folder `data/` — bisa diedit langsung atau via Katyusha:

| File | Fungsi |
|---|---|
| `rules.json` | Parameter trading (min_confidence, min_rr, dll) |
| `logic_rules.json` | Cara deteksi BOS/FVG/IDM/MSS |
| `prompts.json` | Instruksi dan sistem respons per AI |
| `state.json` | State trading persistent (BOS, IDM, MSS, fase) |
| `watchlist.json` | Level harga yang dipantau |

---

## 🖥️ Dashboard

Buka `https://<railway-url>/` untuk:
- Live chat dengan Katyusha
- Toggle Katyusha on/off
- Lihat watchlist aktif + fase per symbol
- History sesi diskusi (loss debrief)
- Tab: Rules, Logic, Prompts, State, Balance

---

## 🔄 Cara Ganti Platform

1. Export env variables dari Railway
2. Pastikan file `data/*.json` di-download/backup
3. Deploy ke platform baru
4. Upload file `data/*.json` ke persistent storage
5. Set env variables

File `data/state.json` menyimpan state trading — kalau hilang, bot mulai fresh dari `h1_scan`.
