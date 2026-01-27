#include "pwm_driver.h"

TIM_HandleTypeDef htim3;

void PWM_Init(void)
{
    TIM_OC_InitTypeDef sConfigOC = {0};
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // Włącz zegar dla TIM3 i GPIOB
    __HAL_RCC_TIM3_CLK_ENABLE();
    __HAL_RCC_GPIOB_CLK_ENABLE();

    // Konfiguracja pinów dla PWM
    // TIM3_CH1 -> PB4 (AF1)
    // TIM3_CH2 -> PB5 (AF1)
    GPIO_InitStruct.Pin = GPIO_PIN_4 | GPIO_PIN_5;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF1_TIM3;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    // Konfiguracja TIM3
    // Zegar APB1 = 48 MHz
    // Częstotliwość PWM = 18 kHz
    // Prescaler = 0 (dzielenie przez 1)
    // Period = (48000000 / 18000) - 1 = 2666 - 1 = 2665
    htim3.Instance = TIM3;
    htim3.Init.Prescaler = 0;
    htim3.Init.CounterMode = TIM_COUNTERMODE_UP;
    htim3.Init.Period = 2665; // 48 MHz / (0+1) / 2666 = ~18 kHz
    htim3.Init.ClockDivision = TIM_CLOCKDIVISION_DIV1;
    htim3.Init.AutoReloadPreload = TIM_AUTORELOAD_PRELOAD_ENABLE;

    if (HAL_TIM_PWM_Init(&htim3) != HAL_OK)
    {
        while (1); // error
    }

    // Konfiguracja kanałów PWM
    sConfigOC.OCMode = TIM_OCMODE_PWM1;
    sConfigOC.Pulse = 0; // Początkowy duty cycle = 0%
    sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
    sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;

    // Kanał 1
    if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
    {
        while (1); // error
    }

    // Kanał 2
    if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
    {
        while (1); // error
    }

    // Start PWM na obu kanałach
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2);
}

void PWM_SetDutyCycle_CH1(uint8_t duty)
{
    if (duty > 100)
        duty = 100;
    uint32_t pulse = (htim3.Init.Period + 1) * duty / 100;
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_1, pulse);
}

void PWM_SetDutyCycle_CH2(uint8_t duty)
{
    if (duty > 100)
        duty = 100;
    uint32_t pulse = (htim3.Init.Period + 1) * duty / 100;
    __HAL_TIM_SET_COMPARE(&htim3, TIM_CHANNEL_2, pulse);
}
