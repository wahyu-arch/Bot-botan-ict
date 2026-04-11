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
