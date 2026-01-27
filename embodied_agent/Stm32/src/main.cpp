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

int main()
{
    // Inicjalizacja systemu
    HAL_Init();
    SystemClock_Config();

    // Konfiguracja serwomechanizmów
    servoSG5010.channel = 0;
    servoSG5010.min_us = 500;
    servoSG5010.max_us = 2300;

    servo1201MG.channel = 1;
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
    Servo_Init(&hi2c1, &servoSG5010, 180);
    Servo_Init(&hi2c1, &servo1201MG, 180);

    // Inicjalizacja SPI
    SPI_Init();

    // Ustaw początkowe wartości PWM
    PWM_SetDutyCycle_CH1(50); // 50% na kanale 1 (PB4)
    PWM_SetDutyCycle_CH2(75); // 75% na kanale 2 (PB5)

    // Uruchom animację serw
    Servo_StartSweep(&servoAnimation, 180, 0, 1);

    // Główna pętla aplikacji
    while (1)
    {
        // Miganie LED
        HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_3);
        HAL_Delay(25);

        // Aktualizacja animacji serw
        uint8_t servo_angle = Servo_UpdateSweep(&servoAnimation);
        Servo_SetAngleWithConfig(&hi2c1, &servoSG5010, servo_angle);
        Servo_SetAngleWithConfig(&hi2c1, &servo1201MG, servo_angle);

        // Obsługa danych z SPI
        if (spi_data_ready)
        {
            spi_data_ready = 0;
            uint8_t received = SPI_GetLastReceivedByte();

            // Przetwarzanie komend z Raspberry Pi
            if (received == 'A')
            {
                PWM_SetDutyCycle_CH1(0);
            }
            else if (received == 'B')
            {
                PWM_SetDutyCycle_CH1(100);
            }
            else if (received >= '0' && received <= '9')
            {
                // Sterowanie PWM cyfrą (0-9 -> 0%-90%)
                PWM_SetDutyCycle_CH1((received - '0') * 10);

                // Sterowanie serwami
                uint8_t angle = (received - '0') * 20;
                Servo_SetAngleWithConfig(&hi2c1, &servoSG5010, angle);
                Servo_SetAngleWithConfig(&hi2c1, &servo1201MG, angle);
            }

            // Opcjonalnie: wyślij odpowiedź do RPi
            // uint8_t response = 0xAA;
            // SPI_SendData(&response, 1);
        }
    }
}