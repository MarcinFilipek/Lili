extern "C"
{
#include "stm32f0xx_hal.h"
#include "pca9685.h"
}

TIM_HandleTypeDef htim3;
SPI_HandleTypeDef hspi1;
I2C_HandleTypeDef hi2c1;

ServoConfig_t servoSG5010, servo1201MG;

// HAL I2C MSP Init callback - wymagane przez HAL
extern "C" void HAL_I2C_MspInit(I2C_HandleTypeDef *hi2c)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    if (hi2c->Instance == I2C1)
    {
        // Włącz zegar dla I2C1 i GPIOF
        __HAL_RCC_I2C1_CLK_ENABLE();
        __HAL_RCC_GPIOF_CLK_ENABLE();

        // PF0 -> I2C1_SDA (AF0) - CN3 pin 5 (A2)
        // PF1 -> I2C1_SCL (AF0) - CN3 pin 9 (A5)
        GPIO_InitStruct.Pin = GPIO_PIN_0 | GPIO_PIN_1;
        GPIO_InitStruct.Mode = GPIO_MODE_AF_OD;
        GPIO_InitStruct.Pull = GPIO_NOPULL; // Zewnętrzne pull-up na module
        GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
        GPIO_InitStruct.Alternate = GPIO_AF1_I2C1;
        HAL_GPIO_Init(GPIOF, &GPIO_InitStruct);
    }
}

// Bufory dla komunikacji SPI
#define SPI_BUFFER_SIZE 64
volatile uint8_t spi_rx_buffer[SPI_BUFFER_SIZE];
volatile uint8_t spi_tx_buffer[SPI_BUFFER_SIZE];
volatile uint8_t spi_rx_index = 0;
volatile uint8_t spi_data_ready = 0;

void SystemClock_Config(void)
{
    RCC_OscInitTypeDef RCC_OscInitStruct = {0};
    RCC_ClkInitTypeDef RCC_ClkInitStruct = {0};

    // Włącz HSI (wewnętrzny oscylator)
    RCC_OscInitStruct.OscillatorType = RCC_OSCILLATORTYPE_HSI;
    RCC_OscInitStruct.HSIState = RCC_HSI_ON;
    RCC_OscInitStruct.HSICalibrationValue = RCC_HSICALIBRATION_DEFAULT;
    RCC_OscInitStruct.PLL.PLLState = RCC_PLL_ON;
    RCC_OscInitStruct.PLL.PLLSource = RCC_PLLSOURCE_HSI;
    RCC_OscInitStruct.PLL.PREDIV = RCC_PREDIV_DIV2;
    RCC_OscInitStruct.PLL.PLLMUL = RCC_PLL_MUL12; // 48 MHz
    if (HAL_RCC_OscConfig(&RCC_OscInitStruct) != HAL_OK)
    {
        while (1)
            ; // error
    }

    RCC_ClkInitStruct.ClockType = RCC_CLOCKTYPE_HCLK | RCC_CLOCKTYPE_SYSCLK | RCC_CLOCKTYPE_PCLK1;
    RCC_ClkInitStruct.SYSCLKSource = RCC_SYSCLKSOURCE_PLLCLK;
    RCC_ClkInitStruct.AHBCLKDivider = RCC_SYSCLK_DIV1;
    RCC_ClkInitStruct.APB1CLKDivider = RCC_HCLK_DIV1;

    if (HAL_RCC_ClockConfig(&RCC_ClkInitStruct, FLASH_LATENCY_1) != HAL_OK)
    {
        while (1)
            ; // error
    }
}

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
        while (1)
            ; // error
    }

    // Konfiguracja kanałów PWM
    sConfigOC.OCMode = TIM_OCMODE_PWM1;
    sConfigOC.Pulse = 0; // Początkowy duty cycle = 0%
    sConfigOC.OCPolarity = TIM_OCPOLARITY_HIGH;
    sConfigOC.OCFastMode = TIM_OCFAST_DISABLE;

    // Kanał 1
    if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_1) != HAL_OK)
    {
        while (1)
            ; // error
    }

    // Kanał 2
    if (HAL_TIM_PWM_ConfigChannel(&htim3, &sConfigOC, TIM_CHANNEL_2) != HAL_OK)
    {
        while (1)
            ; // error
    }

    // Start PWM na obu kanałach
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_1);
    HAL_TIM_PWM_Start(&htim3, TIM_CHANNEL_2);
}

void I2C_Init(void)
{
    // Konfiguracja I2C1 - GPIO będzie skonfigurowane w HAL_I2C_MspInit
    hi2c1.Instance = I2C1;
    hi2c1.Init.Timing = 0x2000090E; // Standard timing dla 100kHz @ 48MHz
    hi2c1.Init.OwnAddress1 = 0;
    hi2c1.Init.AddressingMode = I2C_ADDRESSINGMODE_7BIT;
    hi2c1.Init.DualAddressMode = I2C_DUALADDRESS_DISABLE;
    hi2c1.Init.OwnAddress2 = 0;
    hi2c1.Init.OwnAddress2Masks = I2C_OA2_NOMASK;
    hi2c1.Init.GeneralCallMode = I2C_GENERALCALL_DISABLE;
    hi2c1.Init.NoStretchMode = I2C_NOSTRETCH_DISABLE;

    if (HAL_I2C_Init(&hi2c1) != HAL_OK)
    {
        // Błąd inicjalizacji - zatrzymaj się
        while (1)
            ;
    }

    // Wyłącz filtry - mogą powodować problemy
    HAL_I2CEx_ConfigAnalogFilter(&hi2c1, I2C_ANALOGFILTER_DISABLE);
    HAL_I2CEx_ConfigDigitalFilter(&hi2c1, 0);
}

void SPI_Init(void)
{
    GPIO_InitTypeDef GPIO_InitStruct = {0};

    // Włącz zegar dla SPI1 i GPIOA
    __HAL_RCC_SPI1_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    // Konfiguracja pinów SPI1
    // PA4  -> SPI1_NSS  (Chip Select)
    // PA5  -> SPI1_SCK  (Clock)
    // PA6  -> SPI1_MISO (Master In Slave Out)
    // PA7  -> SPI1_MOSI (Master Out Slave In)
    GPIO_InitStruct.Pin = GPIO_PIN_4 | GPIO_PIN_5 | GPIO_PIN_6 | GPIO_PIN_7;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF0_SPI1;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    // Konfiguracja SPI1 jako Slave
    hspi1.Instance = SPI1;
    hspi1.Init.Mode = SPI_MODE_SLAVE;
    hspi1.Init.Direction = SPI_DIRECTION_2LINES;
    hspi1.Init.DataSize = SPI_DATASIZE_8BIT;
    hspi1.Init.CLKPolarity = SPI_POLARITY_LOW; // CPOL = 0
    hspi1.Init.CLKPhase = SPI_PHASE_1EDGE;     // CPHA = 0 (SPI Mode 0)
    hspi1.Init.NSS = SPI_NSS_HARD_INPUT;
    hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_2; // Nie ma znaczenia dla slave
    hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
    hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
    hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;

    if (HAL_SPI_Init(&hspi1) != HAL_OK)
    {
        while (1)
            ; // error
    }

    // Włącz przerwanie SPI
    HAL_NVIC_SetPriority(SPI1_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(SPI1_IRQn);

    // Wyczyść bufory
    for (int i = 0; i < SPI_BUFFER_SIZE; i++)
    {
        spi_rx_buffer[i] = 0;
        spi_tx_buffer[i] = 0;
    }

    // Rozpocznij odbiór w trybie przerwań
    HAL_SPI_Receive_IT(&hspi1, (uint8_t *)&spi_rx_buffer[0], 1);
}

// Callback wywoływany po odebraniu danych przez SPI
void HAL_SPI_RxCpltCallback(SPI_HandleTypeDef *hspi)
{
    if (hspi->Instance == SPI1)
    {
        spi_rx_index++;
        if (spi_rx_index >= SPI_BUFFER_SIZE)
        {
            spi_rx_index = 0;
        }
        spi_data_ready = 1;

        // Kontynuuj odbiór kolejnego bajtu
        HAL_SPI_Receive_IT(&hspi1, (uint8_t *)&spi_rx_buffer[spi_rx_index], 1);
    }
}

// Callback wywoływany po wysłaniu danych przez SPI
void HAL_SPI_TxCpltCallback(SPI_HandleTypeDef *hspi)
{
    if (hspi->Instance == SPI1)
    {
        // Transmisja zakończona
    }
}

// Funkcja do wysyłania danych przez SPI
void SPI_SendData(uint8_t *data, uint16_t size)
{
    HAL_SPI_Transmit(&hspi1, data, size, 100);
}

// Obsługa przerwania SPI1
extern "C" void SPI1_IRQHandler(void)
{
    HAL_SPI_IRQHandler(&hspi1);
}

// Funkcje pomocnicze do ustawiania duty cycle (0-100%)
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

int main()
{
    HAL_Init();
    SystemClock_Config();

    servoSG5010 = {0, 500, 2500}; // Kanał 0
    servo1201MG = {1, 500, 2500}; // Kanał 1

    // Inicjalizacja PWM
    PWM_Init();

    // Inicjalizacja I2C dla PCA9685
    I2C_Init();

    // Sprawdź czy PCA9685 odpowiada
    HAL_Delay(100); // Daj czas na stabilizację zasilania

    HAL_StatusTypeDef status = HAL_I2C_IsDeviceReady(&hi2c1, 0x40 << 1, 3, 100);
    if (status != HAL_OK)
    {
        // PCA9685 nie odpowiada - migaj szybko LED
        while (1)
        {
            __HAL_RCC_GPIOB_CLK_ENABLE();
            GPIO_InitTypeDef GPIO_InitStruct = {0};
            GPIO_InitStruct.Pin = GPIO_PIN_3;
            GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
            GPIO_InitStruct.Pull = GPIO_NOPULL;
            GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
            HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

            HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_3);
            HAL_Delay(100); // Szybkie miganie = błąd I2C
        }
    }

    // Inicjalizacja sterownika PCA9685
    PCA9685_Init(&hi2c1);

    // Inicjalizacja SPI do komunikacji z Raspberry Pi 5
    SPI_Init();

    // LED na PB3 (opcjonalnie)
    __HAL_RCC_GPIOB_CLK_ENABLE();
    GPIO_InitTypeDef GPIO_InitStruct = {0};
    GPIO_InitStruct.Pin = GPIO_PIN_3;
    GPIO_InitStruct.Mode = GPIO_MODE_OUTPUT_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_LOW;
    HAL_GPIO_Init(GPIOB, &GPIO_InitStruct);

    // Przykład użycia - ustawienie duty cycle
    PWM_SetDutyCycle_CH1(50); // 50% na kanale 1 (PB4)
    PWM_SetDutyCycle_CH2(75); // 75% na kanale 2 (PB5)

    // Przykład sterowania serwem na kanale 0 PCA9685
    // Ustaw początkową pozycję serwa na 90 stopni
    PCA9685_SetServoAngle(&hi2c1, &servoSG5010, 90);
    PCA9685_SetServoAngle(&hi2c1, &servo1201MG, 90);

    uint8_t servo_angle = 90;
    uint8_t direction = 1; // 1 = wzrost, 0 = spadek

    while (1)
    {
        HAL_GPIO_TogglePin(GPIOB, GPIO_PIN_3);
        HAL_Delay(25);

        // Przykładowa animacja serwa - ruch od 0 do 180 stopni
        if (direction)
        {
            servo_angle += 1;
            if (servo_angle >= 180)
            {
                servo_angle = 180;
                direction = 0;
            }
        }
        else
        {
            if (servo_angle >= 1)
                servo_angle -= 1;
            else
                servo_angle = 0;

            if (servo_angle == 0)
            {
                direction = 1;
            }
        }
        PCA9685_SetServoAngle(&hi2c1, &servoSG5010, servo_angle);
        PCA9685_SetServoAngle(&hi2c1, &servo1201MG, servo_angle);

        // Obsługa danych odebranych przez SPI
        if (spi_data_ready)
        {
            spi_data_ready = 0;

            // Przykład: jeśli odebrano 'A', ustaw PWM1 na 0%
            // jeśli odebrano 'B', ustaw PWM1 na 100%
            uint8_t received = spi_rx_buffer[spi_rx_index > 0 ? spi_rx_index - 1 : SPI_BUFFER_SIZE - 1];

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
                // Ustaw PWM na podstawie cyfry (0-9 -> 0%-90%)
                PWM_SetDutyCycle_CH1((received - '0') * 10);

                // Dodatkowo można sterować serwem cyfra -> kąt
                PCA9685_SetServoAngle(&hi2c1, &servoSG5010, (received - '0') * 20);
                PCA9685_SetServoAngle(&hi2c1, &servo1201MG, (received - '0') * 20);
            }

            // Możesz odesłać odpowiedź do RPi
            // uint8_t response = 0xAA;
            // SPI_SendData(&response, 1);
        }

        // Możesz tutaj dynamicznie zmieniać PWM, np:
        // PWM_SetDutyCycle_CH1(jakaś_wartość);
        // PWM_SetDutyCycle_CH2(inna_wartość);
    }
}