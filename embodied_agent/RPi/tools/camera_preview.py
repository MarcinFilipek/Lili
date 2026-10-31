"""Podgląd kamery. 'q' kończy."""

import cv2

from lili.hw.camera import Camera


def main():
    with Camera() as cam:
        while True:
            cv2.imshow("kamera", cam.capture())
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
