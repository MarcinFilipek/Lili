#include "stm32f0xx_hal.h"

void SysTick_Handler(void)
{
    HAL_IncTick();
    HAL_SYSTICK_IRQHandler();
}

// USART1_IRQHandler jest w uart_comm.c
