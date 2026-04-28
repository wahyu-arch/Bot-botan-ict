"""
API Server — expose log diskusi grup AI ke HTML viewer.
Berjalan paralel dengan bot di thread terpisah.
"""

import os
import json
import glob
from datetime import datetime, timezone
from flask import Flask, jsonify, request
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow semua origin agar HTML bisa fetch dari mana saja

# ── In-memory store: bot push chat ke sini secara realtime ──
_live_sessions = []   # list of session dict
_current_session = None  # session yang sedang berjalan
_watchlist = []  # level watchlist aktif


_watchlist_by_symbol: dict = {}

def update_watchlist(items: list, symbol: str = ""):
    """Update watchlist dari bot, per symbol."""
    global _watchlist, _watchlist_by_symbol
    _watchlist = items
    if symbol:
        _watchlist_by_symbol[symbol] = items


def push_message(ai_id: str, nama: str, pesan: str, ronde: int, session_id: str, symbol: str = ""):
    """Dipanggil oleh bot setiap kali ada pesan baru dari AI panel."""
    global _current_session
    if _current_session is None or _current_session["id"] != session_id:
        return
    _current_session["messages"].append({
        "ai": {"ai-1":"arka","ai-2":"nova","ai-3":"zara","arka":"arka","nova":"nova","zara":"zara","yusuf":"yusuf"}.get(ai_id.lower(), ai_id.lower()),
        "nama": nama,
        "pesan": pesan,
        "ronde": ronde,
        "symbol": symbol or _current_session.get("signal", {}).get("symbol", ""),
        "side": "right" if ai_id.lower() == "yusuf" else "left",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })


def start_session(session_id: str, signal: dict, loss_context: str = ""):
    """Dipanggil oleh bot saat diskusi baru dimulai."""
    global _current_session
    _current_session = {
        "id": session_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
        "signal": signal,
        "loss_context": loss_context,
        "messages": [],
        "conclusion": None,
        "status": "ongoing",
    }
    # Simpan ke list (max 50 sesi)
    _live_sessions.insert(0, _current_session)
    if len(_live_sessions) > 50:
        _live_sessions.pop()


def finish_session(conclusion: dict):
    """Dipanggil oleh bot saat diskusi selesai."""
    global _current_session
    if _current_session:
        _current_session["conclusion"] = conclusion
        _current_session["status"] = "done"
        _current_session["finished_at"] = datetime.now(timezone.utc).isoformat()


# ── ENDPOINTS ─────────────────────────────────────────────

@app.route("/")
def index():
    return jsonify({"status": "ok", "service": "ICT AI Panel API"})


@app.route("/api/sessions")
def get_sessions():
    """List semua sesi diskusi (ringkasan)."""
    result = []
    for s in _live_sessions:
        result.append({
            "id": s["id"],
            "started_at": s["started_at"],
            "finished_at": s.get("finished_at"),
            "status": s["status"],
            "loss_context": s.get("loss_context") or None,
            "msg_count": len(s["messages"]),
            "consensus": s["conclusion"].get("consensus") if s.get("conclusion") else None,
            "avg_confidence": s["conclusion"].get("avg_panel_confidence") if s.get("conclusion") else None,
            "signal": s.get("signal", {}),
        })
    return jsonify(result)


@app.route("/api/sessions/<session_id>")
def get_session(session_id):
    """Detail satu sesi — semua pesan + kesimpulan."""
    for s in _live_sessions:
        if s["id"] == session_id:
            return jsonify(s)
    return jsonify({"error": "session not found"}), 404


@app.route("/api/live")
def get_live():
    """Sesi yang sedang berjalan (realtime polling)."""
    if _current_session:
        return jsonify(_current_session)
    return jsonify({"status": "idle", "messages": [], "conclusion": None})


# Pesan-pesan singkat yang masuk ke live feed tanpa sesi formal
# ── Sesi "main" — sesi permanen untuk semua monitoring bot ──────────
_main_session_messages: list = []
_main_session_conclusion: dict = {}

# Status per-symbol dari bot (phase, harga, watchlist count)
_bot_status: dict = {}  # key = symbol

def update_bot_status(symbol: str, phase: str, price: float, wl_count: int):
    """Dipanggil bot setiap siklus untuk update status per symbol."""
    _bot_status[symbol] = {
        "symbol": symbol,
        "phase": phase,
        "price": price,
        "wl_count": wl_count,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }

def push_live_msg(ai: str, nama: str, pesan: str, symbol: str = ""):
    """Push pesan ke sesi 'main' — monitoring BOS, FVG, IDM, entry."""
    global _main_session_messages
    if not pesan:
        return
    _main_session_messages.append({
        "ai": ai, "nama": nama, "pesan": pesan,
        "symbol": symbol,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ronde": 0,
        "side": "right" if ai in ("yusuf","katyusha") else "left",
    })
    _main_session_messages = _main_session_messages[-1000:]


@app.route("/api/live/feed")
def get_live_feed():
    """
    Live feed dengan pagination.
    Query params:
      ?limit=20          — jumlah pesan per halaman (default 20)
      ?before=<idx>      — ambil pesan sebelum index ini (untuk load more)
      ?after=<idx>       — ambil pesan setelah index ini (untuk polling update terbaru)
    Response: {messages, total, has_more, oldest_idx, newest_idx}
    """
    total   = len(_main_session_messages)
    limit   = min(int(request.args.get("limit", 20)), 100)
    before  = request.args.get("before", None)
    after   = request.args.get("after",  None)

    if after is not None:
        # Polling mode: ambil pesan BARU setelah index tertentu
        after_idx = int(after)
        msgs = _main_session_messages[after_idx:]
        return jsonify({
            "messages":   msgs,
            "total":      total,
            "has_more":   after_idx > 0,
            "oldest_idx": after_idx,
            "newest_idx": total,
        })
    elif before is not None:
        # Load more: ambil 20 pesan sebelum index tertentu
        before_idx = int(before)
        start = max(0, before_idx - limit)
        msgs  = _main_session_messages[start:before_idx]
        return jsonify({
            "messages":   msgs,
            "total":      total,
            "has_more":   start > 0,
            "oldest_idx": start,
            "newest_idx": before_idx,
        })
    else:
        # Initial load: 20 pesan terbaru
        start = max(0, total - limit)
        msgs  = _main_session_messages[start:]
        return jsonify({
            "messages":   msgs,
            "total":      total,
            "has_more":   start > 0,
            "oldest_idx": start,
            "newest_idx": total,
        })


@app.route("/api/main")
def get_main_session():
    """
    Sesi main — backward compat: return 20 pesan terbaru + total.
    HTML baru pakai /api/live/feed untuk pagination penuh.
    """
    total = len(_main_session_messages)
    recent = _main_session_messages[-20:]  # max 20 agar tidak berat
    return jsonify({
        "id":         "main",
        "messages":   recent,
        "total_msgs": total,
        "has_more":   total > 20,
        "conclusion": _main_session_conclusion,
        "status":     "active" if total > 0 else "idle",
    })


@app.route("/api/bot_status")
def get_bot_status():
    """Status realtime tiap symbol: fase, harga, jumlah watchlist."""
    return jsonify(list(_bot_status.values()))


# ── Katyusha toggle — persistent ke file ─────────────────
import os as _os, json as _json

_KATY_STATE_FILE = "data/katyusha_state.json"

def _load_katyusha_state() -> bool:
    """
    Baca state dari 3 sumber, urutan prioritas:
    1. File data/katyusha_state.json (paling fresh — dari toggle user)
    2. Env var KATYUSHA_ENABLED (fallback kalau file tidak ada)
    3. Default: True (ON)
    """
    import logging as _log
    _logger = _log.getLogger(__name__)

    # Prioritas 1: file (paling fresh)
    try:
        if _os.path.exists(_KATY_STATE_FILE):
            with open(_KATY_STATE_FILE) as f:
                val = bool(_json.load(f).get("enabled", True))
            _logger.info(f"[KATYUSHA] State loaded dari file: {'ON' if val else 'OFF'}")
            return val
    except Exception as e:
        _logger.warning(f"[KATYUSHA] Gagal baca file state: {e}")

    # Prioritas 2: env var KATYUSHA_ENABLED
    env_val = _os.environ.get("KATYUSHA_ENABLED", "").strip().lower()
    if env_val in ("false", "0", "off", "no"):
        _logger.info("[KATYUSHA] State dari env KATYUSHA_ENABLED=false → OFF")
        return False
    if env_val in ("true", "1", "on", "yes"):
        _logger.info("[KATYUSHA] State dari env KATYUSHA_ENABLED=true → ON")
        return True

    _logger.info("[KATYUSHA] State file & env tidak ada — default ON")
    return True

def _save_katyusha_state(enabled: bool):
    """Simpan ke file. File ini persist kalau Railway volume aktif."""
    import logging as _log
    _logger = _log.getLogger(__name__)
    try:
        _os.makedirs("data", exist_ok=True)
        with open(_KATY_STATE_FILE, "w") as f:
            _json.dump({"enabled": enabled}, f)
        _logger.info(f"[KATYUSHA] State disimpan: {'ON' if enabled else 'OFF'} → {_KATY_STATE_FILE}")
    except Exception as e:
        _logger.warning(f"[KATYUSHA] Gagal simpan state ke file: {e} — state hanya di memory")

# Load saat startup
_katyusha_enabled: bool = _load_katyusha_state()

def is_katyusha_enabled() -> bool:
    return _katyusha_enabled

@app.route("/api/katyusha/toggle", methods=["POST"])
def toggle_katyusha():
    global _katyusha_enabled
    data = request.get_json(silent=True) or {}
    if "enabled" in data:
        _katyusha_enabled = bool(data["enabled"])
    else:
        _katyusha_enabled = not _katyusha_enabled
    _save_katyusha_state(_katyusha_enabled)  # simpan ke file
    return jsonify({"enabled": _katyusha_enabled})

@app.route("/api/katyusha/status")
def katyusha_status():
    return jsonify({"enabled": _katyusha_enabled})


@app.route("/api/ai/<ai_name>")
def get_ai_config(ai_name: str):
    """Baca config AI dari data/ai/<ai_name>.json."""
    import os, json
    path = f"data/ai/{ai_name}.json"
    if not os.path.exists(path):
        return jsonify({"error": f"{ai_name}.json tidak ditemukan"}), 404
    try:
        with open(path, encoding="utf-8") as f:
            return f.read(), 200, {"Content-Type": "application/json"}
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/ai/<ai_name>", methods=["POST"])
def update_ai_config(ai_name: str):
    """Update field di data/ai/<ai_name>.json (dipakai Katyusha)."""
    import os, json
    data = request.get_json(silent=True) or {}
    try:
        from ai_config import save as _ai_save, invalidate_cache
        ok = _ai_save(ai_name, data)
        return jsonify({"ok": ok, "ai": ai_name})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/state")
def get_state():
    """Trading state persistent: BOS, IDM, MSS, reset_count, dll."""
    import os, json
    os.makedirs("data", exist_ok=True)
    try:
        with open("data/state.json") as f:
            return f.read(), 200, {"Content-Type": "application/json"}
    except Exception:
        return '{"active_phase":"h1_scan","reset_count":0}', 200, {"Content-Type": "application/json"}


@app.route("/api/prompts")
@app.route("/api/prompts/")
def get_prompts():
    """Prompt instruksi untuk setiap AI — auto-create default kalau belum ada."""
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/prompts.json"):
        try:
            from prompt_engine import PromptEngine
            PromptEngine()  # constructor auto-create default
        except Exception:
            pass
    try:
        with open("data/prompts.json") as f:
            return f.read(), 200, {"Content-Type": "application/json"}
    except Exception:
        return '{"error": "prompts not found"}', 200, {"Content-Type": "application/json"}


@app.route("/api/rules")
@app.route("/api/rules/")
def get_rules():
    """Rules aktif — auto-create dari default kalau belum ada."""
    import json, os, shutil
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/rules.json"):
        # Coba copy dari rules.json di root (ada di source code)
        for src in ["rules.json", "data/rules.json.default"]:
            if os.path.exists(src):
                shutil.copy(src, "data/rules.json")
                break
        else:
            # Buat minimal default
            default = {"_version": 1, "_update_reason": "Auto-generated", "entry": {"min_confidence": 0.6, "max_confidence_allowed": 0.85}, "tp": {"min_rr": 2.0}, "risk": {"risk_per_trade_pct": 1.0}}
            with open("data/rules.json", "w") as f:
                json.dump(default, f, indent=2)
    try:
        with open("data/rules.json") as f:
            return f.read(), 200, {"Content-Type": "application/json"}
    except Exception as e:
        return json.dumps({"error": str(e)}), 500, {"Content-Type": "application/json"}


@app.route("/api/logic")
@app.route("/api/logic/")
def get_logic():
    """Logic rules aktif — auto-create dari default kalau belum ada."""
    import json, os, shutil
    os.makedirs("data", exist_ok=True)
    if not os.path.exists("data/logic_rules.json"):
        for src in ["data/logic_rules.json.default"]:
            if os.path.exists(src):
                shutil.copy(src, "data/logic_rules.json")
                break
        else:
            default = {"_version": 1, "_update_reason": "Auto-generated", "find_bos_h1": {"method": "swing_break", "swing_left": 8, "swing_right": 8, "lookback": 100}, "find_fvg_h1": {"method": "three_candle_gap", "min_gap_pct": 0.05}, "entry": {"condition": "mss_confirmed", "skip_if_outside_fvg": False}}
            with open("data/logic_rules.json", "w") as f:
                json.dump(default, f, indent=2)
    try:
        with open("data/logic_rules.json") as f:
            return f.read(), 200, {"Content-Type": "application/json"}
    except Exception as e:
        return json.dumps({"error": str(e)}), 500, {"Content-Type": "application/json"}


@app.route("/api/logic/history")
def get_logic_history():
    """History update logic rules."""
    try:
        with open("data/logic_rules_history.json") as f:
            return f.read(), 200, {"Content-Type": "application/json"}
    except Exception:
        return "[]", 200, {"Content-Type": "application/json"}


@app.route("/api/rules/history")
def get_rules_history():
    """History update rules."""
    import json, os
    try:
        with open("data/rules_history.json") as f:
            return f.read(), 200, {"Content-Type": "application/json"}
    except Exception:
        return "[]", 200, {"Content-Type": "application/json"}


@app.route("/api/stats")
def get_stats():
    """Statistik trading: total trade, winrate, trade harian selama 30 hari."""
    import os, json
    from datetime import datetime, timezone, timedelta

    MEMORY_FILE = "data/trade_memory.json"
    try:
        with open(MEMORY_FILE) as f:
            data = json.load(f)
        trades = data.get("trades", [])
    except Exception:
        trades = []

    total = len(trades)
    wins   = sum(1 for t in trades if t.get("result") == "win")
    losses = sum(1 for t in trades if t.get("result") == "loss")
    wr     = round(wins / total * 100, 1) if total else 0

    # PnL total
    total_pnl = sum(t.get("pnl", 0) or 0 for t in trades)

    # Trade harian — 30 hari terakhir
    today = datetime.now(timezone.utc).date()
    daily = {}
    for i in range(30):
        d = (today - timedelta(days=i)).isoformat()
        daily[d] = {"date": d, "total": 0, "wins": 0, "losses": 0, "pnl": 0}

    for t in trades:
        ts = t.get("timestamp") or t.get("entry_time") or ""
        if not ts:
            continue
        try:
            d = ts[:10]  # YYYY-MM-DD
            if d in daily:
                daily[d]["total"] += 1
                if t.get("result") == "win":
                    daily[d]["wins"] += 1
                elif t.get("result") == "loss":
                    daily[d]["losses"] += 1
                daily[d]["pnl"] += t.get("pnl", 0) or 0
        except Exception:
            pass

    # Sort asc untuk chart
    daily_list = sorted(daily.values(), key=lambda x: x["date"])

    return jsonify({
        "total_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": wr,
        "total_pnl": round(total_pnl, 4),
        "daily": daily_list,
        "recent_trades": trades[-20:][::-1],  # 20 trade terakhir, terbaru di atas
    })


# ── User Chat dengan Katyusha ──────────────────────────
_user_chat: list = []   # history chat user ↔ Katyusha

def get_user_chat() -> list:
    return _user_chat

def push_user_chat(role: str, content_text: str):
    """role: 'user' atau 'katyusha'"""
    global _user_chat
    _user_chat.append({
        "role": role,
        "content": content_text,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    })
    _user_chat = _user_chat[-100:]  # max 100 pesan


@app.route("/api/chat", methods=["GET"])
def get_chat():
    return jsonify(_user_chat)


@app.route("/api/chat", methods=["POST"])
def post_chat():
    """User kirim pesan ke Katyusha."""
    data = request.get_json(silent=True) or {}
    msg = data.get("message", "").strip()
    if not msg:
        return jsonify({"error": "message kosong"}), 400
    push_user_chat("user", msg)
    # Bot core akan poll /api/chat/pending dan proses
    return jsonify({"status": "sent", "message": msg})


@app.route("/api/chat/pending")
def get_pending():
    """Bot poll ini untuk ambil pesan user yang belum dibalas."""
    unanswered = [m for m in _user_chat if m["role"] == "user" and not m.get("answered")]
    return jsonify(unanswered)


@app.route("/api/chat/answer", methods=["POST"])
def post_answer():
    """Bot push jawaban Katyusha."""
    data = request.get_json(silent=True) or {}
    answer = data.get("answer", "").strip()
    # Mark semua user messages sebagai answered
    for m in _user_chat:
        if m["role"] == "user":
            m["answered"] = True
    push_user_chat("katyusha", answer)
    return jsonify({"status": "ok"})


@app.route("/api/balance")
def get_balance():
    """Saldo Bybit dari trade_executor."""
    try:
        import sys, os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from trade_executor import TradeExecutor
        executor = TradeExecutor(paper_mode=os.getenv("PAPER_TRADING","true").lower()=="true")
        balance = executor.get_account_balance()
        syms = os.getenv("TRADING_SYMBOLS", os.getenv("TRADING_SYMBOL","BTCUSDT")); return jsonify({"balance_usdt": balance, "symbols": syms})
    except Exception as e:
        return jsonify({"error": str(e)[:100]}), 500


@app.route("/api/watchlist")
def get_watchlist():
    """Watchlist level aktif — semua symbol digabung."""
    if _watchlist_by_symbol:
        all_items = []
        for items in _watchlist_by_symbol.values():
            all_items.extend(items)
        return jsonify(all_items)
    return jsonify(_watchlist)


@app.route("/api/watchlist/<symbol>")
def get_watchlist_symbol(symbol):
    """Watchlist untuk satu symbol spesifik."""
    return jsonify(_watchlist_by_symbol.get(symbol.upper(), []))


@app.route("/api/symbols")
def get_symbols():
    """Symbol aktif dan status watchlist-nya."""
    import os
    raw = os.getenv("TRADING_SYMBOLS", os.getenv("TRADING_SYMBOL", "BTCUSDT"))
    symbols = [s.strip() for s in raw.split(",") if s.strip()]
    status = {}
    for sym in symbols:
        wl = _watchlist_by_symbol.get(sym, [])
        active = [w for w in wl if not w.get("triggered")]
        status[sym] = {"watchlist_active": len(active), "total": len(wl)}
    return jsonify({"symbols": symbols, "status": status})


@app.route("/api/latest")
def get_latest():
    """Sesi terakhir yang selesai."""
    for s in _live_sessions:
        if s["status"] == "done":
            return jsonify(s)
    return jsonify({"status": "no_session"})


def run_server():
    """Jalankan Flask di port 8080 (Railway expose port ini)."""
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
