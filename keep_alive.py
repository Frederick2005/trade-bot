"""
Keeps Render free tier awake by running a tiny HTTP server.
UptimeRobot pings / every 5 minutes — this responds with 200 OK.
"""
from http.server import HTTPServer, BaseHTTPRequestHandler
import threading
from loguru import logger


class HealthHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        from app.state import state
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        msg = (
            f"crypto-bot alive | "
            f"balance=${state.balance:.2f} | "
            f"open_trades={state.open_trade_count()} | "
            f"paused={state.is_paused} | "
            f"binance={'ok' if state.binance_connected else 'down'}"
        )
        self.wfile.write(msg.encode())

    def log_message(self, format, *args):
        pass  # silence noisy request logs


def start_health_server(port: int = 8080) -> None:
    server = HTTPServer(("0.0.0.0", port), HealthHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    logger.info(f"Health server started on port {port}")