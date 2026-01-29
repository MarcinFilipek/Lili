#include "servo_control.h"

// Tablica przechowująca aktualne pozycje serw
uint8_t servo_current_angles[MAX_SERVOS] = {0};

void Servo_Init(I2C_HandleTypeDef *hi2c, ServoConfig_t *servo, uint8_t initial_angle)
{
    // Ustaw początkową pozycję serwa
    PCA9685_SetServoAngle(hi2c, servo, initial_angle);
    servo_current_angles[servo->channel] = initial_angle;
}

void Servo_SetAngleWithConfig(I2C_HandleTypeDef *hi2c, ServoConfig_t *servo, uint8_t angle)
{
    // Wykorzystuje strukturę ServoConfig_t z min_us i max_us
    PCA9685_SetServoAngle(hi2c, servo, angle);
    servo_current_angles[servo->channel] = angle;
}

uint8_t Servo_GetCurrentAngle(uint8_t channel)
{
    if (channel < MAX_SERVOS)
        return servo_current_angles[channel];
    return 0;
}

void Servo_StartSweep(ServoAnimation_t *animation, uint8_t start_angle, uint8_t end_angle, uint8_t step)
{
    animation->current_angle = start_angle;
    animation->target_angle = end_angle;
    animation->step = step;
    animation->active = 1;
}

uint8_t Servo_UpdateSweep(ServoAnimation_t *animation)
{
    if (!animation->active)
        return animation->current_angle;

    if (animation->current_angle < animation->target_angle)
    {
        animation->current_angle += animation->step;
        if (animation->current_angle >= animation->target_angle)
        {
            animation->current_angle = animation->target_angle;
            // Odwróć kierunek
            uint8_t temp = animation->current_angle;
            animation->current_angle = animation->target_angle;
            animation->target_angle = temp;
        }
    }
    else
    {
        if (animation->current_angle >= animation->step)
            animation->current_angle -= animation->step;
        else
            animation->current_angle = 0;

        if (animation->current_angle <= animation->target_angle)
        {
            animation->current_angle = animation->target_angle;
            // Odwróć kierunek
            uint8_t temp = animation->current_angle;
            animation->current_angle = animation->target_angle;
            animation->target_angle = temp;
        }
    }

    return animation->current_angle;
}
