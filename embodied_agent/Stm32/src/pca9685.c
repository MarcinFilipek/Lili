#include "pca9685.h"
#include <math.h>

// Funkcja pomocnicza do zapisu do rejestru PCA9685
static HAL_StatusTypeDef PCA9685_WriteReg(I2C_HandleTypeDef *hi2c, uint8_t reg, uint8_t value)
{
    uint8_t data[2];
    data[0] = reg;
    data[1] = value;
    return HAL_I2C_Master_Transmit(hi2c, PCA9685_ADDRESS << 1, data, 2, HAL_MAX_DELAY);
}

// Funkcja pomocnicza do odczytu z rejestru PCA9685
static HAL_StatusTypeDef PCA9685_ReadReg(I2C_HandleTypeDef *hi2c, uint8_t reg, uint8_t *value)
{
    HAL_StatusTypeDef status;
    status = HAL_I2C_Master_Transmit(hi2c, PCA9685_ADDRESS << 1, &reg, 1, HAL_MAX_DELAY);
    if (status != HAL_OK) return status;
    return HAL_I2C_Master_Receive(hi2c, PCA9685_ADDRESS << 1, value, 1, HAL_MAX_DELAY);
}

// Inicjalizacja PCA9685
void PCA9685_Init(I2C_HandleTypeDef *hi2c)
{
    // Reset
    PCA9685_WriteReg(hi2c, PCA9685_MODE1, MODE1_RESTART);
    HAL_Delay(10);
    
    // Ustaw tryb: wyłącz tryb uśpienia, włącz AI (auto-increment)
    PCA9685_WriteReg(hi2c, PCA9685_MODE1, MODE1_AI | MODE1_ALLCALL);
    HAL_Delay(1);
    
    // Ustaw MODE2
    PCA9685_WriteReg(hi2c, PCA9685_MODE2, 0x04);  // Totem pole outputs
    
    // Ustaw częstotliwość PWM
    PCA9685_SetPWMFreq(hi2c, PCA9685_PWM_FREQ);
}

// Ustawienie częstotliwości PWM
void PCA9685_SetPWMFreq(I2C_HandleTypeDef *hi2c, float freq)
{
    uint8_t prescale, oldmode, newmode;
    float prescaleval;
    
    // Oblicz wartość prescalera
    // prescaleval = 25MHz / (4096 * freq) - 1
    prescaleval = 25000000.0f;  // Zegar wewnętrzny 25 MHz
    prescaleval /= 4096.0f;     // 12-bit
    prescaleval /= freq;
    prescaleval -= 1.0f;
    
    prescale = (uint8_t)(prescaleval + 0.5f);
    
    // Odczytaj obecny tryb
    PCA9685_ReadReg(hi2c, PCA9685_MODE1, &oldmode);
    
    // Przejdź w tryb uśpienia (wymagane do zmiany prescalera)
    newmode = (oldmode & 0x7F) | MODE1_SLEEP;
    PCA9685_WriteReg(hi2c, PCA9685_MODE1, newmode);
    
    // Ustaw prescaler
    PCA9685_WriteReg(hi2c, PCA9685_PRESCALE, prescale);
    
    // Wyjdź z trybu uśpienia
    PCA9685_WriteReg(hi2c, PCA9685_MODE1, oldmode);
    HAL_Delay(1);
    
    // Restart
    PCA9685_WriteReg(hi2c, PCA9685_MODE1, oldmode | MODE1_RESTART);
}

// Ustawienie PWM na kanale (wartości 0-4095)
void PCA9685_SetPWM(I2C_HandleTypeDef *hi2c, uint8_t channel, uint16_t on, uint16_t off)
{
    uint8_t data[5];
    
    data[0] = PCA9685_LED0_ON_L + 4 * channel;
    data[1] = on & 0xFF;
    data[2] = on >> 8;
    data[3] = off & 0xFF;
    data[4] = off >> 8;
    
    HAL_I2C_Master_Transmit(hi2c, PCA9685_ADDRESS << 1, data, 5, HAL_MAX_DELAY);
}

// Ustawienie serwa według kąta (0-180 stopni)
void PCA9685_SetServoAngle(I2C_HandleTypeDef *hi2c, ServoConfig_t* servo, uint8_t angle)
{
    if (angle > 180) angle = 180;
    
    uint16_t pulse_us = servo->min_us + ((uint16_t)angle * (servo->max_us - servo->min_us)) / 180;
    PCA9685_SetServoPulse(hi2c, servo->channel, pulse_us);
}

// Ustawienie serwa według długości impulsu (w mikrosekundach)
void PCA9685_SetServoPulse(I2C_HandleTypeDef *hi2c, uint8_t channel, uint16_t pulse_us)
{
    // Dla 50Hz: jeden cykl = 20ms = 20000us
    // 4096 kroków na 20000us → ~4.88us na krok
    // off_value = pulse_us / 4.88
    uint16_t off_value = (uint16_t)((float)pulse_us * 4096.0f / 20000.0f);
    PCA9685_SetPWM(hi2c, channel, 0, off_value);
}
