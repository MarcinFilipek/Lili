#ifndef PCA9685_H
#define PCA9685_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "stm32f0xx_hal.h"

// Adresy rejestrów PCA9685
#define PCA9685_ADDRESS 0x40 // Domyślny adres I2C (7-bit)
#define PCA9685_MODE1 0x00
#define PCA9685_MODE2 0x01
#define PCA9685_SUBADR1 0x02
#define PCA9685_SUBADR2 0x03
#define PCA9685_SUBADR3 0x04
#define PCA9685_PRESCALE 0xFE
#define PCA9685_LED0_ON_L 0x06
#define PCA9685_LED0_ON_H 0x07
#define PCA9685_LED0_OFF_L 0x08
#define PCA9685_LED0_OFF_H 0x09
#define PCA9685_ALL_LED_ON_L 0xFA
#define PCA9685_ALL_LED_ON_H 0xFB
#define PCA9685_ALL_LED_OFF_L 0xFC
#define PCA9685_ALL_LED_OFF_H 0xFD

// Bity rejestrów
#define MODE1_RESTART 0x80
#define MODE1_SLEEP 0x10
#define MODE1_ALLCALL 0x01
#define MODE1_AI 0x20 // Auto-Increment

// Częstotliwość PWM (Hz)
#define PCA9685_PWM_FREQ 50 // Typowa dla serw

    typedef struct
    {
        uint8_t channel; // Kanał PCA9685 (0-15)
        uint16_t min_us; // Minimalny impuls w mikrosekundach
        uint16_t max_us; // Maksymalny impuls w mikrosekundach
    } ServoConfig_t;

    // Funkcje API
    void PCA9685_Init(I2C_HandleTypeDef *hi2c);
    void PCA9685_SetPWMFreq(I2C_HandleTypeDef *hi2c, float freq);
    void PCA9685_SetPWM(I2C_HandleTypeDef *hi2c, uint8_t channel, uint16_t on, uint16_t off);
    void PCA9685_SetServoAngle(I2C_HandleTypeDef *hi2c, ServoConfig_t *servo, uint8_t angle);
    void PCA9685_SetServoPulse(I2C_HandleTypeDef *hi2c, uint8_t channel, uint16_t pulse_us);

#ifdef __cplusplus
}
#endif

#endif // PCA9685_H
