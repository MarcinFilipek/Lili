import spidev
import time
import sys


def main():
    # Initialize SPI
    spi = spidev.SpiDev()
    bus = 0
    device = 0

    try:
        spi.open(bus, device)

        # Settings for RPi 5 / general SPI usage
        # Max speed in Hz (e.g., 500000 = 500kHz)
        spi.max_speed_hz = 500000
        # SPI mode (0, 1, 2, 3)
        spi.mode = 0

        print(f"SPI Initialized on Bus {bus}, Device {device}")
        print("Enter 'A' to send value A, 'B' to send value B.")
        print("Press Ctrl+C or enter 'Q' to quit.")

        while True:
            user_input = input("Enter command (A/B): ").strip().upper()

            if user_input == "Q":
                break

            data_to_send = []
            if user_input == "A":
                # Example: send ASCII 'A' or some specific value
                data_to_send = [ord("A")]
                print("Sending 'A'...")
            elif user_input == "B":
                # Example: send ASCII 'B' or some specific value
                data_to_send = [ord("B")]
                print("Sending 'B'...")
            else:
                print("Invalid command. Please enter A or B.")
                continue

            if data_to_send:
                # xfer2 sends and receives data. It keeps CS active between bytes if list > 1
                resp = spi.xfer2(data_to_send)
                print(f"Sent: {data_to_send}, Received: {resp}")

    except IOError as e:
        print(
            f"Error: Could not access SPI device. Ensure SPI is enabled and you have permissions.\n{e}"
        )
    except KeyboardInterrupt:
        print("\nExiting...")
    finally:
        spi.close()
        print("SPI Closed.")


if __name__ == "__main__":
    main()
