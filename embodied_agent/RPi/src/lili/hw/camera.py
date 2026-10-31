import time

WIDTH = 320
HEIGHT = 240
FPS = 30


class Camera:
    """Picamera2, surowe klatki. Przetwarzanie należy do warstwy agenta.

    gray=True: kanał Y (luminancja) z YUV420, bez konwersji kolorów.
    gray=False: format "RGB888", który w picamera2 daje w pamięci kolejność BGR (zgodną z OpenCV).
    """

    def __init__(self, width=WIDTH, height=HEIGHT, gray=True, fps=FPS):
        from picamera2 import Picamera2

        self.width, self.height, self.gray, self.fps = width, height, gray, fps
        self.cam = Picamera2()
        frame_us = int(1_000_000 / fps)
        conf = self.cam.create_video_configuration(
            main={"format": "YUV420" if gray else "RGB888", "size": (width, height)},
            controls={"FrameDurationLimits": (frame_us, frame_us)},
        )
        self.cam.configure(conf)
        self.cam.start()

    def capture(self):
        """Zwraca (obraz, t, t_sensor): t z time.monotonic_ns() po odebraniu klatki, t_sensor z metadanych."""
        request = self.cam.capture_request()
        try:
            image = request.make_array("main")
            metadata = request.get_metadata()
        finally:
            request.release()
        t = time.monotonic_ns()
        if self.gray:
            image = image[: self.height, : self.width].copy()
        return image, t, metadata.get("SensorTimestamp", 0)

    def lock_exposure(self, warmup_s=2.0):
        """Po ustaleniu się automatyki zamraża ekspozycję i balans bieli. Zwraca ustawione wartości.

        Zmiany jasności w trakcie sesji wyglądają dla przepływu optycznego jak ruch.
        """
        time.sleep(warmup_s)
        metadata = self.cam.capture_metadata()
        controls = {
            "AeEnable": False,
            "AwbEnable": False,
            "ExposureTime": metadata["ExposureTime"],
            "AnalogueGain": metadata["AnalogueGain"],
        }
        if "ColourGains" in metadata:
            controls["ColourGains"] = metadata["ColourGains"]
        self.cam.set_controls(controls)
        return controls

    def info(self):
        return {
            "model": self.cam.camera_properties.get("Model"),
            "width": self.width,
            "height": self.height,
            "gray": self.gray,
            "fps": self.fps,
            "sensor": self.cam.camera_configuration().get("sensor"),
        }

    def close(self):
        self.cam.stop()
        self.cam.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
