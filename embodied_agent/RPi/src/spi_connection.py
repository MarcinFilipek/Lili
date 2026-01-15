import spidev


class SPIConnection:
    def __init__(self):
        self.spi = spidev.SpiDev()

        try:
             self.spi.open(0, 0)
            self.spi.max_speed_hz = 500000
        except Exception as e:
            print(f"Error initializing SPI: {e}")

    def send(self, data):
        try:
            self.spi.xfer2(data)
        except Exception as e:
            print(f"Error sending data: {e}")
