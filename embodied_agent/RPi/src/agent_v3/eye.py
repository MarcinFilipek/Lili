from picamera2 import Picamera2
import cv2

width = 640
height = 480


class Eye:
    def __init__(self):
        self.eye = Picamera2()
        cam_conf = self.eye.create_video_configuration(
            main={"format": "RGB888", "size": (width, height)}
        )
        self.eye.configure(cam_conf)
        self.eye.start()

    def capture(self, gray=True, blur=True):
        frame = self.eye.capture_array("main")

        if gray:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        if blur:
            # frame = cv2.GaussianBlur(frame, (77, 77), 0)
            frame = cv2.pyrDown(cv2.pyrDown(frame))
            frame = cv2.pyrUp(cv2.pyrUp(frame))

        return frame

    def close(self):
        self.eye.stop()
