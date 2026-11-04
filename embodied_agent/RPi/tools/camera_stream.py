"""Podgląd kamery na żywo w przeglądarce (MJPEG przez HTTP), do ustawiania sceny przez SSH.

RPi:  python tools/camera_stream.py
Mac:  ssh -N -L 8000:localhost:8000 uzume@raspberrypi
      potem w przeglądarce http://localhost:8000

Domyślnie serwer słucha tylko na localhost RPi, więc dostęp jest wyłącznie przez tunel SSH.
--host 0.0.0.0 udostępnia go w całej sieci lokalnej (bez hasła). Ctrl+C kończy.
"""

import argparse
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import cv2

from lili.hw.camera import Camera

PAGE = b"""<!doctype html><html><head><meta charset="utf-8"><title>Lili kamera</title>
<style>body{margin:0;background:#111;display:flex;justify-content:center;align-items:center;height:100vh}
img{height:96vh;image-rendering:pixelated}</style></head>
<body><img src="/stream.mjpg"></body></html>"""


def draw_guides(image):
    """Krzyż w punkcie głównym i linia horyzontu: pomoc przy ustawianiu obiektów i listwy."""
    h, w = image.shape[:2]
    color = (0, 255, 255) if image.ndim == 3 else 255
    cv2.line(image, (w // 2, 0), (w // 2, h - 1), color, 1)
    cv2.line(image, (0, h // 2), (w - 1, h // 2), color, 1)
    return image


class LatestFrame:
    """Ostatnia klatka jako JPEG, odczytywana przez wiele połączeń naraz."""

    def __init__(self):
        self.jpeg = None
        self.count = 0
        self.condition = threading.Condition()

    def publish(self, jpeg):
        with self.condition:
            self.jpeg = jpeg
            self.count += 1
            self.condition.notify_all()

    def wait_newer(self, count, timeout=2.0):
        with self.condition:
            self.condition.wait_for(lambda: self.count > count, timeout)
            return self.jpeg, self.count


def capture_loop(camera, latest, stop, guides, quality):
    while not stop.is_set():
        image, _, _ = camera.capture()
        if guides:
            image = draw_guides(image.copy())
        ok, jpeg = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if ok:
            latest.publish(jpeg.tobytes())


def make_handler(latest):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(PAGE)
            elif self.path == "/stream.mjpg":
                self.send_response(200)
                self.send_header("Cache-Control", "no-cache")
                self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                self.end_headers()
                count = 0
                try:
                    while True:
                        jpeg, count = latest.wait_newer(count)
                        if jpeg is None:
                            continue
                        self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                        self.wfile.write(f"Content-Length: {len(jpeg)}\r\n\r\n".encode() + jpeg + b"\r\n")
                except (BrokenPipeError, ConnectionResetError):
                    pass  # przeglądarka zamknięta
            else:
                self.send_error(404)

        def log_message(self, *args):
            pass  # bez wpisu na każde połączenie

    return Handler


def serve(camera, host, port, guides=True, quality=80):
    latest, stop = LatestFrame(), threading.Event()
    worker = threading.Thread(target=capture_loop, args=(camera, latest, stop, guides, quality), daemon=True)
    worker.start()
    server = ThreadingHTTPServer((host, port), make_handler(latest))
    server.daemon_threads = True
    print(f"Podgląd na http://{host}:{port} (Ctrl+C kończy)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
        stop.set()
        worker.join(timeout=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1", help="0.0.0.0 = cała sieć lokalna")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--color", action="store_true", help="kolor zamiast szarości z nagrań")
    ap.add_argument("--size", type=int, nargs=2, default=[320, 240], metavar=("W", "H"))
    ap.add_argument("--no-guides", action="store_true", help="bez krzyża w środku kadru")
    args = ap.parse_args()

    with Camera(width=args.size[0], height=args.size[1], gray=not args.color) as camera:
        serve(camera, args.host, args.port, guides=not args.no_guides)


if __name__ == "__main__":
    main()
