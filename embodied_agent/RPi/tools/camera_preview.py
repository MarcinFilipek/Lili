"""Podgląd kamery w trybie nagrywania (320x240, szarość). 'q' kończy."""

import cv2

from lili.hw.camera import Camera


def main():
    with Camera() as cam:
        while True:
            image, _, _ = cam.capture()
            cv2.imshow("kamera", image)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
