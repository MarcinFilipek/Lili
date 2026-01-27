#ifndef PWM_DRIVER_H
#define PWM_DRIVER_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "stm32f0xx_hal.h"

    // Eksport globalnego handle'a timera
    extern TIM_HandleTypeDef htim3;

    // Funkcje inicjalizacji i sterowania PWM
    void PWM_Init(void);
    void PWM_SetDutyCycle_CH1(uint8_t duty);
    void PWM_SetDutyCycle_CH2(uint8_t duty);

#ifdef __cplusplus
}
#endif

#endif // PWM_DRIVER_H
