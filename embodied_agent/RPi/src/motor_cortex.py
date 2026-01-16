from spi_connection import SPIConnection


class MotorCortex:
    def __init__(self):
        self.spi = SPIConnection()  # To STM32
        self.action = [0, 0, 0, 0, 1]

    def execute(self, action):
        self.action = action
        # command = 1
        # self.spi.send([command])
