from spi_connection import SPIConnection


class MotorCortex:
    def __init__(self):
        self.spi = SPIConnection()  # To STM32

    def execute(self, command):
        self.spi.send([command])
