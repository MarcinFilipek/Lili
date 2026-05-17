import serial
import time
import sys

class UARTConnection:
    def __init__(self):
        self.START_BYTE = 0xAA
        self.CMD_SET_POSITION = 0x01
        self.CMD_GET_POSITION = 0x02
        
        try:
            self.ser = serial.Serial('/dev/ttyAMA0', 115200, timeout=0.1)
        except Exception as e:
            print(f"Error initializing UART: {e}")

    def _calculate_checksum(self, start, cmd, channel, data):
        return start ^ cmd ^ channel ^ data

    def _map_from_byte(self, byte_value):
        # STM32 zwraca [0, 180]° → [-90, 90]°
        return float(byte_value) - 90.0

    def set_servo_position(self, channel, byte_value):
        """
        Set servo position on STM32 via UART.
        """
        checksum = self._calculate_checksum(
            self.START_BYTE, self.CMD_SET_POSITION, channel, byte_value
        )
        request = bytearray([self.START_BYTE, self.CMD_SET_POSITION, channel, byte_value, checksum])
        
        try:
            self.ser.reset_input_buffer()
            self.ser.write(request)
            
            # W UART nie musimy wysyłać dummy bytes. Po prostu czekamy na 5 bajtów odpowiedzi.
            response = self.ser.read(5)
            
            if len(response) == 5 and response[0] == self.START_BYTE:
                return list(response)
            else:
                raise Exception("Timeout lub błędna odpowiedź z STM32!")
        except Exception as e:
            raise

    def read_servo_position(self, channel):
        """
        Request and read servo position from STM32 via UART.
        """
        checksum = self._calculate_checksum(
            self.START_BYTE, self.CMD_GET_POSITION, channel, 0x00
        )
        request = bytearray([self.START_BYTE, self.CMD_GET_POSITION, channel, 0x00, checksum])
        
        try:
            self.ser.reset_input_buffer()
            self.ser.write(request)
            
            response = self.ser.read(5)
            
            if len(response) != 5:
                raise Exception(f"Timeout UART! Odebrano {len(response)} bajtów.")
                
            if response[0] != self.START_BYTE:
                raise Exception(f"Invalid start byte: {response[0]:#x} (expected {self.START_BYTE:#x})")
                
            expected_checksum = self._calculate_checksum(response[0], response[1], response[2], response[3])
            
            if response[4] != expected_checksum:
                raise Exception(f"Checksum mismatch: got {response[4]:#x}, expected {expected_checksum:#x}")
                
            data_byte = response[3]
            position = self._map_from_byte(data_byte)
            return position
            
        except Exception as e:
            raise
