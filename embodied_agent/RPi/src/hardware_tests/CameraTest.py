import cv2
from picamera2 import Picamera2

width = 640
height = 480

picam = Picamera2()
cam_conf = picam.create_video_configuration(main={"format": "RGB888", "size": (width, height)})
picam.configure(cam_conf)

picam.start()

def main():
    while True:
        frame = picam.capture_array()
        cv2.imshow("kamera: ", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()
    picam.stop()



if __name__ == "__main__":
    main()