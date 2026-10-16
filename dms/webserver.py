"""Phat dashboard len trinh duyet (http://localhost:PORT) dang MJPEG, chi dung thu vien chuan.

Xu ly van chay tren may nay; trinh duyet chi nhan anh dashboard da ve san.
Chi lang nghe 127.0.0.1 nen may khac trong mang khong xem duoc.
"""
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

PAGE = """<!doctype html>
<html lang="vi"><head><meta charset="utf-8"><title>VinFast DriverGuard</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
  html,body{margin:0;height:100%;background:#0e141e;color:#e8eef6;font-family:"Segoe UI",Arial,sans-serif}
  main{display:flex;flex-direction:column;align-items:center;justify-content:center;min-height:100%;padding:12px;box-sizing:border-box}
  img{max-width:100%;max-height:calc(100vh - 56px);border:1px solid #2c384c;border-radius:6px}
  p{color:#8c9cb2;font-size:13px;margin:8px 0 0}
</style></head>
<body><main>
  <img src="/stream" alt="Dashboard VinFast DriverGuard">
  <p>Đang chạy trên máy này (localhost). Dừng: nhấn Ctrl+C trong terminal.</p>
</main></body></html>
"""


class FrameHub:
    """Giu khung JPEG moi nhat; vong lap camera ghi, cac ket noi trinh duyet doc."""

    def __init__(self):
        self.cond = threading.Condition()
        self.jpeg, self.seq = None, 0

    def publish(self, img):
        ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 80])
        if ok:
            with self.cond:
                self.jpeg, self.seq = buf.tobytes(), self.seq + 1
                self.cond.notify_all()

    def wait_next(self, last_seq, timeout=2.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq, timeout=timeout)
            return self.jpeg, self.seq


def serve(hub, port=8000):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass  # khong in log moi request

        def do_GET(self):
            if self.path == "/":
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            elif self.path == "/stream":
                self.send_response(200)
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                seq = -1
                try:
                    while True:
                        jpeg, seq = hub.wait_next(seq)
                        if jpeg is None:
                            time.sleep(0.05)
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode())
                        self.wfile.write(jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                    pass  # trinh duyet dong tab
            else:
                self.send_error(404)

    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    return httpd
