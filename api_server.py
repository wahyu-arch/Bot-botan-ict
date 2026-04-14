"""
API Server — expose log diskusi grup AI ke HTML viewer.
Berjalan paralel dengan bot di thread terpisah.
"""

import os
import json
import glob
from datetime import datetime, timezone
from flask import Flask, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Allow semua origin agar HTML bisa fetch dari mana saja

# ── In-memory store: bot push chat ke sini secara realtime ──
_live_sessions = []   # list of session dict
_current_session = None  # session yang sedang berjalan
_watchlist = []  # level watchlist aktif


def update_watchlist(items: list):
    """Update watchlist dari bot."""
    global _watchlist
    _watchlist = items


def push_message(ai_id: str, nama: str, pesan: str, ronde: int, session_id: str):
    """Dipanggil oleh bot setiap kali ada pesan baru dari AI panel."""
    global _current_session
    if _current_session is None or _current_session["id"] != session_id:
        return
    _current_session["messages"].append({
        "ai": {"ai-1":"arka","ai-2":"nova","ai-3":"zara","arka":"arka","nova":"nova","zara":"zara","yusuf":"yusuf"}.get(ai_id.lower(), ai_id.lower()),
        "nama": nama,
        "pesan": pesan,
        "ronde": ronde,
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


@app.route("/api/rules")
@app.route("/api/rules/")
def get_rules():
    """Rules aktif saat ini."""
    import json, os
    os.makedirs("data", exist_ok=True)
    try:
        with open("data/rules.json") as f:
            return f.read(), 200, {"Content-Type": "application/json"}
    except Exception:
        # File belum ada — kembalikan default kosong
        return '{"_version": 1, "_update_reason": "File not created yet — bot belum jalan"}', 200, {"Content-Type": "application/json"}


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


@app.route("/api/watchlist")
def get_watchlist():
    """Watchlist level aktif."""
    return jsonify(_watchlist)


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
