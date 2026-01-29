#ifndef SERVO_CONTROL_H
#define SERVO_CONTROL_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "stm32f0xx_hal.h"
#include "pca9685.h"

#define MAX_SERVOS 16

    // Struktura animacji serwa
    typedef struct
    {
        uint8_t current_angle;
        uint8_t target_angle;
        uint8_t step;
        uint8_t active;
    } ServoAnimation_t;

    // Tablica przechowująca aktualne kąty serw
    extern uint8_t servo_current_angles[MAX_SERVOS];

    // Funkcje sterowania serwami
    void Servo_Init(I2C_HandleTypeDef *hi2c, ServoConfig_t *servo, uint8_t initial_angle);
    void Servo_SetAngleWithConfig(I2C_HandleTypeDef *hi2c, ServoConfig_t *servo, uint8_t angle);
    uint8_t Servo_GetCurrentAngle(uint8_t channel);
    void Servo_StartSweep(ServoAnimation_t *animation, uint8_t start_angle, uint8_t end_angle, uint8_t step);
    uint8_t Servo_UpdateSweep(ServoAnimation_t *animation);

#ifdef __cplusplus
}
#endif

#endif // SERVO_CONTROL_H
