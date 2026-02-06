extern "C"
{
#include "stm32f0xx_hal.h"
#include "system_config.h"
#include "pwm_driver.h"
#include "spi_comm.h"
#include "servo_control.h"
#include "pca9685.h"
}

// Konfiguracje serwomechanizmów
ServoConfig_t servoSG5010, servo1201MG;

// Animacja serwa
ServoAnimation_t servoAnimation;

// Funkcja obsługi ramek SPI
void ProcessSPIFrame(SPI_Frame_t *frame)
{
    SPI_Frame_t response;
    response.channel = frame->channel;

    switch (frame->cmd_type)
    {
    case SPI_CMD_SET_SERVO:
    {
        // Ustaw kąt serwa
        uint8_t angle = frame->data;

        if (frame->channel == 1)
        {
            Servo_SetAngleWithConfig(&hi2c1, &servoSG5010, angle);
        }
        else if (frame->channel == 0)
        {
            Servo_SetAngleWithConfig(&hi2c1, &servo1201MG, angle);
        }

        // Wyślij potwierdzenie
        response.cmd_type = SPI_CMD_ACK;
        response.data = angle;
        SPI_PrepareResponse(&response);
        break;
    }

    case SPI_CMD_GET_SERVO:
    {
        // Odczytaj aktualny kąt serwa
        uint8_t current_angle = Servo_GetCurrentAngle(frame->channel);

        response.cmd_type = SPI_CMD_ACK;
        response.data = current_angle;
        SPI_PrepareResponse(&response);
        break;
    }

    case SPI_CMD_SET_PWM:
    {
        // Ustaw PWM
        uint8_t duty = frame->data;

        if (frame->channel == 1)
            PWM_SetDutyCycle_CH1(duty);
        else if (frame->channel == 2)
            PWM_SetDutyCycle_CH2(duty);

        response.cmd_type = SPI_CMD_ACK;
        response.data = duty;
        SPI_PrepareResponse(&response);
        break;
    }

    case SPI_CMD_PING:
    {
        // Test połączenia
        response.cmd_type = SPI_CMD_ACK;
        response.data = 0x55; // Signature
        SPI_PrepareResponse(&response);
        break;
    }

    case SPI_CMD_RESET:
    {
        // Reset animacji
        servoAnimation.active = 0;

        response.cmd_type = SPI_CMD_ACK;
        response.data = 0x00;
        SPI_PrepareResponse(&response);
        break;
    }

    default:
    {
        // Nieznana komenda
        response.cmd_type = SPI_CMD_NACK;
        response.data = 0xFF;
        SPI_PrepareResponse(&response);
        break;
    }
    }
}

int main()
{
    // Inicjalizacja systemu
    HAL_Init();
    SystemClock_Config();

    // Konfiguracja serwomechanizmów
    servoSG5010.channel = 1;
    servoSG5010.min_us = 1000; // 500us
    servoSG5010.max_us = 1600; // 2300us

    servo1201MG.channel = 0;
    servo1201MG.min_us = 500;
    servo1201MG.max_us = 2500;

    // Inicjalizacja peryferiów
    PWM_Init();
    System_I2C_Init();
    System_GPIO_Init();

    // Sprawdź czy PCA9685 odpowiada
    HAL_Delay(100);
    HAL_StatusTypeDef status = HAL_I2C_IsDeviceReady(&hi2c1, 0x40 << 1, 3, 100);
    if (status != HAL_OK)
    {
        System_ErrorHandler(); // PCA9685 nie odpowiada
    }

    // Inicjalizacja sterownika PCA9685
    PCA9685_Init(&hi2c1);

    // Inicjalizacja serwomechanizmów
    Servo_Init(&hi2c1, &servoSG5010, 90);
    Servo_Init(&hi2c1, &servo1201MG, 90);

    // Inicjalizacja SPI
    SPI_Init();

    // Ustaw początkowe wartości PWM
    PWM_SetDutyCycle_CH1(50); // 50% na kanale 1 (PB4)
    PWM_SetDutyCycle_CH2(75); // 75% na kanale 2 (PB5)

    // Uruchom animację serw
    // Servo_StartSweep(&servoAnimation, 180, 0, 1);

    // Główna pętla aplikacji
    while (1)
    {
        // Miganie LED
        HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_3);
        HAL_Delay(25);

        // Aktualizacja animacji serw (można wyłączyć przez SPI)
        // if (servoAnimation.active)
        // {
        //     uint8_t servo_angle = Servo_UpdateSweep(&servoAnimation);
        //     Servo_SetAngleWithConfig(&hi2c1, &servoSG5010, servo_angle);
        //     Servo_SetAngleWithConfig(&hi2c1, &servo1201MG, servo_angle);
        // }

        // Obsługa ramek SPI
        if (frame_ready)
        {
            frame_ready = 0;
            ProcessSPIFrame(&current_frame);
        }
    }
}

// import spidev
// import time

// spi = spidev.SpiDev()
// spi.open(0, 0)  # Bus 0, Device 0
// spi.max_speed_hz = 1000000

// def calculate_checksum(start, cmd, channel, data):
//     return start ^ cmd ^ channel ^ data

// def send_frame(cmd_type, channel, data):
//     start_byte = 0xAA
//     checksum = calculate_checksum(start_byte, cmd_type, channel, data)
//     frame = [start_byte, cmd_type, channel, data, checksum]
//     response = spi.xfer2(frame)
//     time.sleep(0.01)
//     return response

// # Ustaw servo 0 na 90 stopni
// send_frame(0x01, 0, 90)

// # Odczytaj pozycję servo 0
// response = send_frame(0x02, 0, 0x00)

// # Test połączenia
// send_frame(0x05, 0, 0x00)