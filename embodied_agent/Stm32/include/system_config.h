#ifndef SYSTEM_CONFIG_H
#define SYSTEM_CONFIG_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "stm32f0xx_hal.h"

    // Eksport globalnych handleów (definiowane w system_config.c)
    extern I2C_HandleTypeDef hi2c1;

    // Funkcje konfiguracyjne systemu
    void SystemClock_Config(void);
    void System_I2C_Init(void);
    void System_GPIO_Init(void);
    void System_ErrorHandler(void);

#ifdef __cplusplus
}
#endif

#endif // SYSTEM_CONFIG_H
