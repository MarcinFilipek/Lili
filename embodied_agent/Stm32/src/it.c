#include "stm32f0xx_hal.h"

void SysTick_Handler(void)
{
    HAL_IncTick();
    HAL_SYSTICK_IRQHandler();
}

// Obsługa przerwania SPI1 została przeniesiona do spi_comm.c
