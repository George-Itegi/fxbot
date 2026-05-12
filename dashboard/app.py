"""
Deriv Bot — Live Dashboard v2
============================
FastAPI-based dashboard for monitoring bot performance in real-time.
v2: Market/Volatility column, Duration/Ticks column, Load More pagination.
"""

import asyncio
import json
import sys
import time
from pathlib import Path
from typing import Optional

# Add parent dir to path so imports work when running from dashboard/
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, WebSocket
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles

from config import DASHBOARD_PORT, DASHBOARD_HOST, SYMBOLS
from utils.logger import setup_logger

logger = setup_logger("dashboard.app")


# ─── Dashboard HTML ───
HTML_RESPONSE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Deriv Over/Under Bot</title>
    <style>
        * { margin: 0; padding: 0; box-sizing: border-box; }
        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: #0a0a0f;
            color: #e0e0e0;
            min-height: 100vh;
        }
        .header {
            background: linear-gradient(135deg, #1a1a2e, #16213e);
            padding: 20px 30px;
            border-bottom: 1px solid #2a2a4a;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .header h1 { font-size: 24px; color: #00d4ff; }
        .header .status {
            padding: 6px 16px;
            border-radius: 20px;
            font-size: 14px;
            font-weight: 600;
        }
        .status.running { background: #00c853; color: #000; }
        .status.stopped { background: #ff5252; color: #fff; }
        .status.paper { background: #ffab00; color: #000; }
        .status.live { background: #ff5252; color: #fff; }
        .grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            padding: 20px 30px;
        }
        .card {
            background: #12121f;
            border: 1px solid #2a2a4a;
            border-radius: 12px;
            padding: 20px;
        }
        .card .label {
            font-size: 12px;
            color: #888;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-bottom: 8px;
        }
        .card .value {
            font-size: 28px;
            font-weight: 700;
        }
        .value.positive { color: #00e676; }
        .value.negative { color: #ff5252; }
        .value.neutral { color: #ffd740; }
        .section {
            padding: 20px 30px;
        }
        .section h2 {
            font-size: 18px;
            color: #00d4ff;
            margin-bottom: 16px;
            padding-bottom: 8px;
            border-bottom: 1px solid #2a2a4a;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .section h2 .trade-count {
            font-size: 13px;
            color: #888;
            font-weight: 400;
        }
        .trades-table {
            width: 100%;
            border-collapse: collapse;
        }
        .trades-table th, .trades-table td {
            padding: 10px 12px;
            text-align: left;
            border-bottom: 1px solid #1a1a2e;
            font-size: 13px;
            white-space: nowrap;
        }
        .trades-table th {
            color: #888;
            text-transform: uppercase;
            font-size: 11px;
            letter-spacing: 1px;
        }
        .trades-table tr:hover {
            background: #1a1a2e;
        }
        .win { color: #00e676; }
        .loss { color: #ff5252; }
        .market-badge {
            display: inline-block;
            padding: 2px 8px;
            border-radius: 4px;
            font-size: 11px;
            font-weight: 600;
        }
        .market-1s { background: #1a237e; color: #82b1ff; }
        .market-vol { background: #1b5e20; color: #69f0ae; }
        .market-other { background: #3e2723; color: #ffab91; }
        .duration-badge {
            display: inline-block;
            padding: 2px 6px;
            border-radius: 4px;
            background: #263238;
            color: #80cbc4;
            font-size: 11px;
            font-weight: 600;
        }
        .direction-over { color: #69f0ae; }
        .direction-under { color: #ff8a80; }
        .drift-alert {
            background: #ff5252;
            color: white;
            padding: 10px 20px;
            border-radius: 8px;
            margin: 10px 30px;
            display: none;
            font-weight: 600;
        }
        .drift-alert.active { display: block; }
        .load-more-container {
            text-align: center;
            padding: 16px 0;
        }
        .load-more-btn {
            background: #1a1a2e;
            color: #00d4ff;
            border: 1px solid #2a2a4a;
            padding: 10px 32px;
            border-radius: 8px;
            cursor: pointer;
            font-size: 14px;
            font-weight: 600;
            transition: all 0.2s;
        }
        .load-more-btn:hover {
            background: #2a2a4a;
            border-color: #00d4ff;
        }
        .load-more-btn:disabled {
            color: #555;
            border-color: #333;
            cursor: default;
        }
        .footer {
            text-align: center;
            padding: 20px;
            color: #555;
            font-size: 12px;
        }
        .loading-spinner {
            display: inline-block;
            width: 14px;
            height: 14px;
            border: 2px solid #2a2a4a;
            border-top: 2px solid #00d4ff;
            border-radius: 50%;
            animation: spin 0.8s linear infinite;
            margin-right: 8px;
            vertical-align: middle;
        }
        @keyframes spin { to { transform: rotate(360deg); } }
    </style>
</head>
<body>
    <div class="header">
        <h1>Deriv Over/Under Bot</h1>
        <div>
            <span id="mode-badge" class="status paper">PAPER</span>
            <span id="run-badge" class="status stopped">STOPPED</span>
        </div>
    </div>

    <div id="drift-alert" class="drift-alert">
        Concept drift detected - model may be unreliable
    </div>

    <div class="grid">
        <div class="card">
            <div class="label">Bankroll</div>
            <div class="value" id="bankroll">$100.00</div>
        </div>
        <div class="card">
            <div class="label">Total P&amp;L</div>
            <div class="value neutral" id="pnl">$0.00</div>
        </div>
        <div class="card">
            <div class="label">ROI</div>
            <div class="value neutral" id="roi">0.0%</div>
        </div>
        <div class="card">
            <div class="label">Win Rate</div>
            <div class="value" id="winrate">0.0%</div>
        </div>
        <div class="card">
            <div class="label">Trades</div>
            <div class="value" id="trades">0</div>
        </div>
        <div class="card">
            <div class="label">Model Accuracy</div>
            <div class="value" id="modelacc">0.0%</div>
        </div>
        <div class="card">
            <div class="label">Markets</div>
            <div class="value" id="symbol">13</div>
        </div>
        <div class="card">
            <div class="label">Uptime</div>
            <div class="value" id="uptime">0m</div>
        </div>
    </div>

    <div class="section">
        <h2>
            <span>Recent Trades</span>
            <span class="trade-count" id="trade-count">Showing 0 of 0</span>
        </h2>
        <div style="overflow-x:auto;">
            <table class="trades-table">
                <thead>
                    <tr>
                        <th>#</th>
                        <th>Time</th>
                        <th>Market</th>
                        <th>Direction</th>
                        <th>Barrier</th>
                        <th>Duration</th>
                        <th>Stake</th>
                        <th>Confidence</th>
                        <th>EV</th>
                        <th>Result</th>
                        <th>P&amp;L</th>
                    </tr>
                </thead>
                <tbody id="trades-body">
                    <tr><td colspan="11" style="text-align:center;color:#555;">Waiting for trades...</td></tr>
                </tbody>
            </table>
        </div>
        <div class="load-more-container">
            <button class="load-more-btn" id="load-more-btn" onclick="loadMoreTrades()">
                Load More Trades
            </button>
        </div>
    </div>

    <div class="footer">
        Deriv Over/Under Bot v7 | Updates every 2s
    </div>

    <script>
        let ws;
        let reconnectTimer;
        let currentLimit = 20;
        let totalTrades = 0;

        function connectWS() {
            const protocol = location.protocol === 'https:' ? 'wss:' : 'ws:';
            ws = new WebSocket(protocol + '//' + location.host + '/ws');

            ws.onopen = function() {
                console.log('Dashboard WS connected');
            };

            ws.onmessage = function(event) {
                try {
                    const state = JSON.parse(event.data);
                    updateUI(state);
                } catch(e) {
                    console.error('WS parse error:', e);
                }
            };

            ws.onclose = function() {
                console.log('WS disconnected, reconnecting in 3s...');
                clearTimeout(reconnectTimer);
                reconnectTimer = setTimeout(connectWS, 3000);
            };

            ws.onerror = function() {
                ws.close();
            };
        }

        connectWS();

        function formatTime(ts) {
            if (!ts) return '-';
            var d = new Date(ts * 1000);
            return d.toLocaleTimeString();
        }

        function formatUptime(seconds) {
            var h = Math.floor(seconds / 3600);
            var m = Math.floor((seconds % 3600) / 60);
            return h > 0 ? h + 'h ' + m + 'm' : m + 'm';
        }

        function getMarketClass(symbol) {
            if (symbol && symbol.startsWith('1HZ')) return 'market-1s';
            if (symbol && symbol.startsWith('R_')) return 'market-vol';
            return 'market-other';
        }

        function getMarketName(symbol) {
            var names = {
                '1HZ10V': 'Vol 10 (1s)', '1HZ15V': 'Vol 15 (1s)',
                '1HZ25V': 'Vol 25 (1s)', '1HZ30V': 'Vol 30 (1s)',
                '1HZ50V': 'Vol 50 (1s)', '1HZ75V': 'Vol 75 (1s)',
                '1HZ90V': 'Vol 90 (1s)', '1HZ100V': 'Vol 100 (1s)',
                'R_10': 'Vol 10', 'R_25': 'Vol 25', 'R_50': 'Vol 50',
                'R_75': 'Vol 75', 'R_100': 'Vol 100'
            };
            return names[symbol] || symbol;
        }

        function formatDirection(dir) {
            if (dir === 'DIGITOVER') return '<span class="direction-over">Over</span>';
            if (dir === 'DIGITUNDER') return '<span class="direction-under">Under</span>';
            return dir;
        }

        function updateUI(state) {
            if (!state) return;

            document.getElementById('bankroll').textContent = '$' + state.bankroll.toFixed(2);

            var pnl = state.total_pnl || 0;
            var pnlEl = document.getElementById('pnl');
            pnlEl.textContent = (pnl >= 0 ? '+' : '') + '$' + pnl.toFixed(2);
            pnlEl.className = 'value ' + (pnl > 0 ? 'positive' : pnl < 0 ? 'negative' : 'neutral');

            var roi = state.roi || 0;
            var roiEl = document.getElementById('roi');
            roiEl.textContent = (roi >= 0 ? '+' : '') + roi.toFixed(1) + '%';
            roiEl.className = 'value ' + (roi > 0 ? 'positive' : roi < 0 ? 'negative' : 'neutral');

            document.getElementById('winrate').textContent = (state.win_rate || 0).toFixed(1) + '%';
            document.getElementById('trades').textContent = state.total_trades || 0;
            document.getElementById('modelacc').textContent = (state.model_accuracy || 0).toFixed(1) + '%';
            document.getElementById('symbol').textContent = state.markets_count || 13;
            document.getElementById('uptime').textContent = formatUptime(state.uptime_seconds || 0);

            totalTrades = state.total_trades || 0;

            var runBadge = document.getElementById('run-badge');
            runBadge.textContent = state.running ? 'RUNNING' : 'STOPPED';
            runBadge.className = 'status ' + (state.running ? 'running' : 'stopped');

            var modeBadge = document.getElementById('mode-badge');
            modeBadge.textContent = (state.mode || 'paper').toUpperCase();
            modeBadge.className = 'status ' + (state.mode === 'live' ? 'live' : 'paper');

            document.getElementById('drift-alert').className =
                'drift-alert ' + (state.drift_active ? 'active' : '');
        }

        function renderTrades(trades) {
            if (!trades || trades.length === 0) return;
            var tbody = document.getElementById('trades-body');
            var html = trades.map(function(t) {
                var won = t.won;
                var cls = won ? 'win' : 'loss';
                var result = won ? 'WIN' : 'LOSS';
                var pl = won ? t.payout : -Number(t.stake);
                var symbol = t.symbol || 'N/A';
                var duration = t.duration || 5;
                var ev = Number(t.expected_value || 0);

                return '<tr>' +
                    '<td>' + t.trade_id + '</td>' +
                    '<td>' + formatTime(t.timestamp) + '</td>' +
                    '<td><span class="market-badge ' + getMarketClass(symbol) + '">' + getMarketName(symbol) + '</span></td>' +
                    '<td>' + formatDirection(t.direction) + '</td>' +
                    '<td>' + t.barrier + '</td>' +
                    '<td><span class="duration-badge">' + duration + 't</span></td>' +
                    '<td>$' + Number(t.stake).toFixed(2) + '</td>' +
                    '<td>' + (Number(t.confidence) * 100).toFixed(0) + '%</td>' +
                    '<td>' + (ev >= 0 ? '+' : '') + ev.toFixed(3) + '</td>' +
                    '<td class="' + cls + '">' + result + '</td>' +
                    '<td class="' + cls + '">' + (pl >= 0 ? '+' : '') + '$' + Math.abs(Number(pl)).toFixed(2) + '</td>' +
                    '</tr>';
            }).join('');
            tbody.innerHTML = html;

            // Update trade count
            document.getElementById('trade-count').textContent =
                'Showing ' + trades.length + ' of ' + totalTrades;

            // Update load more button
            var btn = document.getElementById('load-more-btn');
            if (trades.length >= totalTrades) {
                btn.disabled = true;
                btn.textContent = 'All trades loaded';
            } else {
                btn.disabled = false;
                btn.textContent = 'Load More Trades (' + (totalTrades - trades.length) + ' remaining)';
            }
        }

        function loadMoreTrades() {
            currentLimit += 20;
            var btn = document.getElementById('load-more-btn');
            btn.innerHTML = '<span class="loading-spinner"></span>Loading...';
            btn.disabled = true;

            fetch('/api/trades?limit=' + currentLimit).then(function(r) { return r.json(); }).then(function(trades) {
                renderTrades(trades);
            }).catch(function() {
                btn.textContent = 'Load More Trades';
                btn.disabled = false;
            });
        }

        // Poll REST endpoints for stats + trades
        setInterval(function() {
            try {
                // Fetch computed state from MySQL
                fetch('/api/state').then(function(r) { return r.json(); }).then(function(state) {
                    if (state) updateUI(state);
                });

                fetch('/api/trades?limit=' + currentLimit).then(function(r) { return r.json(); }).then(function(trades) {
                    if (trades && trades.length > 0) {
                        renderTrades(trades);
                    }
                });
            } catch (e) {}
        }, 2000);
    </script>
</body>
</html>
"""


app = FastAPI(title="Deriv Over/Under Bot Dashboard", version="2.0.0")

# Shared state — updated by the bot's main loop
_bot_state = {
    "running": False,
    "symbol": "",
    "mode": "paper",
    "bankroll": 0,
    "total_pnl": 0,
    "roi": 0,
    "total_trades": 0,
    "wins": 0,
    "losses": 0,
    "win_rate": 0,
    "daily_pnl": 0,
    "consecutive_losses": 0,
    "circuit_breaker": False,
    "model_accuracy": 0,
    "model_version": 0,
    "drift_active": False,
    "last_signal": None,
    "last_trade": None,
    "ticks_processed": 0,
    "uptime_seconds": 0,
    "start_time": time.time(),
    "markets_count": 13,
}

# WebSocket clients for live updates
_ws_clients: list[WebSocket] = []


def update_state(**kwargs):
    """Update shared dashboard state (called by bot main loop)."""
    _bot_state.update(kwargs)
    _bot_state["uptime_seconds"] = time.time() - _bot_state["start_time"]


@app.get("/", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard HTML."""
    return HTML_RESPONSE


@app.get("/api/state")
async def get_state():
    """Get current bot state — computed from MySQL so it works standalone."""
    try:
        from utils.metrics import PerformanceTracker
        tracker = PerformanceTracker()
        s = tracker.summary()
        return {
            "running": True if tracker.trade_count > 0 else False,
            "symbol": "Multi-market",
            "mode": _bot_state.get("mode", "live"),
            "bankroll": 100.0 + s["total_pnl"],
            "total_pnl": s["total_pnl"],
            "roi": (s["total_pnl"] / 100.0 * 100) if s["total_trades"] > 0 else 0,
            "total_trades": s["total_trades"],
            "win_rate": s["win_rate"],
            "model_accuracy": 0,
            "model_version": 0,
            "drift_active": False,
            "uptime_seconds": time.time() - _bot_state["start_time"],
            "markets_count": _bot_state.get("markets_count", 13),
        }
    except Exception as e:
        logger.error(f"Failed to compute state: {e}")
        return _bot_state


@app.get("/api/trades")
async def get_trades(limit: int = 20):
    """Get recent trades from database."""
    try:
        from utils.metrics import PerformanceTracker
        tracker = PerformanceTracker()
        return tracker.get_recent_trades(limit)
    except Exception as e:
        logger.error(f"Failed to fetch trades: {e}")
        return []


@app.get("/api/performance")
async def get_performance():
    """Get performance summary."""
    try:
        from utils.metrics import PerformanceTracker
        tracker = PerformanceTracker()
        return tracker.summary()
    except Exception as e:
        logger.error(f"Failed to fetch performance: {e}")
        return {"error": str(e)}


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket for real-time state updates."""
    await ws.accept()
    _ws_clients.append(ws)

    try:
        while True:
            await asyncio.sleep(2)
            await ws.send_json(_bot_state)
    except Exception:
        pass
    finally:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


def run_dashboard():
    """Start the dashboard server (called from main bot)."""
    import uvicorn
    logger.info(f"Dashboard starting on {DASHBOARD_HOST}:{DASHBOARD_PORT}")
    uvicorn.run(app, host=DASHBOARD_HOST, port=DASHBOARD_PORT, log_level="info")


if __name__ == "__main__":
    print(f"\n  Dashboard: http://localhost:{DASHBOARD_PORT}\n")
    run_dashboard()
