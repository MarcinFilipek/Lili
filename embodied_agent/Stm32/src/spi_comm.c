#include "spi_comm.h"
#include <string.h>

SPI_HandleTypeDef hspi1;

// Bufory dla full-duplex SPI
volatile uint8_t spi_rx_frame[5] = {0};
volatile uint8_t spi_tx_frame[5] = {0xAA, 0x00, 0x00, 0x00, 0x00}; // Domyślna odpowiedź

// Parser ramek
SPI_Frame_t current_frame;
volatile uint8_t frame_ready = 0;
volatile uint8_t response_pending = 0;

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
    hspi1.Init.BaudRatePrescaler = SPI_BAUDRATEPRESCALER_2;
    hspi1.Init.FirstBit = SPI_FIRSTBIT_MSB;
    hspi1.Init.TIMode = SPI_TIMODE_DISABLE;
    hspi1.Init.CRCCalculation = SPI_CRCCALCULATION_DISABLE;

    if (HAL_SPI_Init(&hspi1) != HAL_OK)
    {
        while (1); // error
    }

    // Włącz przerwanie SPI
    HAL_NVIC_SetPriority(SPI1_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(SPI1_IRQn);

    // Rozpocznij ciągłą komunikację full-duplex (5 bajtów)
    HAL_SPI_TransmitReceive_IT(&hspi1, 
                               (uint8_t*)spi_tx_frame, 
                               (uint8_t*)spi_rx_frame, 
                               5);
}

uint8_t SPI_CalculateChecksum(SPI_Frame_t *frame)
{
    return frame->start_byte ^ frame->cmd_type ^ frame->channel ^ frame->data;
}

uint8_t SPI_IsFrameValid(SPI_Frame_t *frame)
{
    if (frame->start_byte != SPI_FRAME_START_BYTE)
        return 0;
    
    uint8_t calculated = SPI_CalculateChecksum(frame);
    return (calculated == frame->checksum);
}

void SPI_PrepareResponse(SPI_Frame_t *frame)
{
    frame->start_byte = SPI_FRAME_START_BYTE;
    frame->checksum = SPI_CalculateChecksum(frame);
    
    // Skopiuj do bufora TX
    spi_tx_frame[0] = frame->start_byte;
    spi_tx_frame[1] = frame->cmd_type;
    spi_tx_frame[2] = frame->channel;
    spi_tx_frame[3] = frame->data;
    spi_tx_frame[4] = frame->checksum;
    
    response_pending = 1;
}

// Callback wywoływany po zakończeniu TransmitReceive
void HAL_SPI_TxRxCpltCallback(SPI_HandleTypeDef *hspi)
{
    if (hspi->Instance == SPI1)
    {
        // Skopiuj odebraną ramkę
        memcpy((void*)&current_frame, (void*)spi_rx_frame, 5);
        
        // Sprawdź poprawność ramki
        if (SPI_IsFrameValid(&current_frame))
        {
            frame_ready = 1;  // Sygnalizuj main loop
        }
        
        // Jeśli nie ma nowej odpowiedzi, wyślij dummy frame
        if (!response_pending)
        {
            spi_tx_frame[0] = 0xAA;
            spi_tx_frame[1] = 0x00;
            spi_tx_frame[2] = 0x00;
            spi_tx_frame[3] = 0x00;
            spi_tx_frame[4] = 0x00;
        }
        else
        {
            response_pending = 0;
        }
        
        // Restart komunikacji - kolejne 5 bajtów
        HAL_SPI_TransmitReceive_IT(&hspi1, 
                                   (uint8_t*)spi_tx_frame, 
                                   (uint8_t*)spi_rx_frame, 
                                   5);
    }
}

// Obsługa przerwania SPI1
void SPI1_IRQHandler(void)
{
    HAL_SPI_IRQHandler(&hspi1);
}
