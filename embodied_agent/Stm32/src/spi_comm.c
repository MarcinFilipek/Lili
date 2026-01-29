#include "spi_comm.h"

SPI_HandleTypeDef hspi1;
volatile uint8_t spi_rx_buffer[SPI_BUFFER_SIZE];
volatile uint8_t spi_tx_buffer[SPI_BUFFER_SIZE];
volatile uint8_t spi_rx_index = 0;
volatile uint8_t spi_data_ready = 0;

// Parser ramek
SPI_Frame_t current_frame;
SPI_ParserState parser_state = SPI_STATE_IDLE;
volatile uint8_t frame_ready = 0;

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

    // Wyczyść bufory
    for (int i = 0; i < SPI_BUFFER_SIZE; i++)
    {
        spi_rx_buffer[i] = 0;
        spi_tx_buffer[i] = 0;
    }

    // Reset parsera
    parser_state = SPI_STATE_IDLE;
    frame_ready = 0;

    // Rozpocznij odbiór w trybie przerwań
    HAL_SPI_Receive_IT(&hspi1, (uint8_t *)&spi_rx_buffer[0], 1);
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

void SPI_ProcessByte(uint8_t byte)
{
    switch (parser_state)
    {
        case SPI_STATE_IDLE:
            if (byte == SPI_FRAME_START_BYTE)
            {
                current_frame.start_byte = byte;
                parser_state = SPI_STATE_WAIT_CMD;
            }
            break;
            
        case SPI_STATE_WAIT_CMD:
            current_frame.cmd_type = byte;
            parser_state = SPI_STATE_WAIT_CHANNEL;
            break;
            
        case SPI_STATE_WAIT_CHANNEL:
            current_frame.channel = byte;
            parser_state = SPI_STATE_WAIT_DATA;
            break;
            
        case SPI_STATE_WAIT_DATA:
            current_frame.data = byte;
            parser_state = SPI_STATE_WAIT_CHECKSUM;
            break;
            
        case SPI_STATE_WAIT_CHECKSUM:
            current_frame.checksum = byte;
            parser_state = SPI_STATE_IDLE;
            
            // Sprawdź poprawność ramki
            if (SPI_IsFrameValid(&current_frame))
            {
                frame_ready = 1;
            }
            break;
            
        default:
            parser_state = SPI_STATE_IDLE;
            break;
    }
}

void SPI_SendFrame(SPI_Frame_t *frame)
{
    frame->start_byte = SPI_FRAME_START_BYTE;
    frame->checksum = SPI_CalculateChecksum(frame);
    
    uint8_t buffer[5];
    buffer[0] = frame->start_byte;
    buffer[1] = frame->cmd_type;
    buffer[2] = frame->channel;
    buffer[3] = frame->data;
    buffer[4] = frame->checksum;
    
    HAL_SPI_Transmit(&hspi1, buffer, 5, 100);
}

void SPI_SendData(uint8_t *data, uint16_t size)
{
    HAL_SPI_Transmit(&hspi1, data, size, 100);
}

uint8_t SPI_GetLastReceivedByte(void)
{
    return spi_rx_buffer[spi_rx_index > 0 ? spi_rx_index - 1 : SPI_BUFFER_SIZE - 1];
}

// Callback wywoływany po odebraniu danych przez SPI
void HAL_SPI_RxCpltCallback(SPI_HandleTypeDef *hspi)
{
    if (hspi->Instance == SPI1)
    {
        uint8_t received = spi_rx_buffer[spi_rx_index];
        
        // Przetwórz bajt przez parser ramek
        SPI_ProcessByte(received);
        
        spi_rx_index++;
        if (spi_rx_index >= SPI_BUFFER_SIZE)
        {
            spi_rx_index = 0;
        }
        
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

// Obsługa przerwania SPI1
void SPI1_IRQHandler(void)
{
    HAL_SPI_IRQHandler(&hspi1);
}
