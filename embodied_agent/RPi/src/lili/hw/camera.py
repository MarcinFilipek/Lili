WIDTH = 640
HEIGHT = 480


class Camera:
    """Picamera2, surowe klatki. Przetwarzanie należy do warstwy agenta.

    Format "RGB888" w picamera2 daje w pamięci kolejność BGR (zgodną z OpenCV).
    """

    def __init__(self, width=WIDTH, height=HEIGHT):
        from picamera2 import Picamera2

        self.cam = Picamera2()
        conf = self.cam.create_video_configuration(main={"format": "RGB888", "size": (width, height)})
        self.cam.configure(conf)
        self.cam.start()

    def capture(self):
        return self.cam.capture_array("main")

    def close(self):
        self.cam.stop()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
