#include "uart_comm.h"
#include <string.h>

UART_HandleTypeDef huart1;

UART_CmdQueue_t uart_cmd_queue = {.head = 0, .tail = 0};

// Zmienne dla automatu stanów odbioru
volatile uint8_t rx_byte = 0;
volatile uint8_t rx_buffer[5] = {0};
volatile uint8_t rx_index = 0;
volatile uint8_t rx_state = 0; // 0: Czeka na START, 1: Odbiera dane

static void UART_Queue_Push(const UART_Frame_t *frame)
{
    uint8_t next_head = (uart_cmd_queue.head + 1) & (UART_CMD_QUEUE_SIZE - 1);
    if (next_head != uart_cmd_queue.tail)
    {
        uart_cmd_queue.frames[uart_cmd_queue.head] = *frame;
        uart_cmd_queue.head = next_head;
    }
}

uint8_t UART_Queue_Pop(UART_Frame_t *out_frame)
{
    if (uart_cmd_queue.tail == uart_cmd_queue.head)
        return 0; // pusta

    *out_frame = uart_cmd_queue.frames[uart_cmd_queue.tail];
    uart_cmd_queue.tail = (uart_cmd_queue.tail + 1) & (UART_CMD_QUEUE_SIZE - 1);
    return 1;
}

uint8_t UART_CalculateChecksum(UART_Frame_t *frame)
{
    return frame->start_byte ^ frame->cmd_type ^ frame->channel ^ frame->data;
}

void UART_Init(void)
{
    __HAL_RCC_USART1_CLK_ENABLE();
    __HAL_RCC_GPIOA_CLK_ENABLE();

    GPIO_InitTypeDef GPIO_InitStruct = {0};
    // PA9 (TX = D1), PA10 (RX = D0)
    GPIO_InitStruct.Pin = GPIO_PIN_9 | GPIO_PIN_10;
    GPIO_InitStruct.Mode = GPIO_MODE_AF_PP;
    GPIO_InitStruct.Pull = GPIO_NOPULL;
    GPIO_InitStruct.Speed = GPIO_SPEED_FREQ_HIGH;
    GPIO_InitStruct.Alternate = GPIO_AF1_USART1;
    HAL_GPIO_Init(GPIOA, &GPIO_InitStruct);

    huart1.Instance = USART1;
    huart1.Init.BaudRate = 115200;
    huart1.Init.WordLength = UART_WORDLENGTH_8B;
    huart1.Init.StopBits = UART_STOPBITS_1;
    huart1.Init.Parity = UART_PARITY_NONE;
    huart1.Init.Mode = UART_MODE_TX_RX;
    huart1.Init.HwFlowCtl = UART_HWCONTROL_NONE;
    huart1.Init.OverSampling = UART_OVERSAMPLING_16;
    
    if (HAL_UART_Init(&huart1) != HAL_OK)
    {
        while (1); // Błąd krytyczny
    }

    // Włączenie przerwań
    HAL_NVIC_SetPriority(USART1_IRQn, 0, 0);
    HAL_NVIC_EnableIRQ(USART1_IRQn);

    // Rozpoczęcie nasłuchu na pierwszy bajt
    HAL_UART_Receive_IT(&huart1, (uint8_t*)&rx_byte, 1);
}

void USART1_IRQHandler(void)
{
    HAL_UART_IRQHandler(&huart1);
}

void HAL_UART_RxCpltCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1)
    {
        if (rx_state == 0) // Szukanie bajtu synchronizacji
        {
            if (rx_byte == UART_FRAME_START_BYTE)
            {
                rx_buffer[0] = rx_byte;
                rx_index = 1;
                rx_state = 1; // Znaleziono, przejdź do odbioru
            }
        }
        else // Odbieranie kolejnych bajtów
        {
            rx_buffer[rx_index++] = rx_byte;
            
            if (rx_index == 5) // Ramka kompletna
            {
                UART_Frame_t frame;
                frame.start_byte = rx_buffer[0];
                frame.cmd_type = rx_buffer[1];
                frame.channel = rx_buffer[2];
                frame.data = rx_buffer[3];
                frame.checksum = rx_buffer[4];

                // Weryfikacja sumy kontrolnej
                if (UART_CalculateChecksum(&frame) == frame.checksum)
                {
                    UART_Queue_Push(&frame);
                }
                
                // Reset automatu stanów
                rx_state = 0;
                rx_index = 0;
            }
        }
        
        // Zawsze nasłuchuj następnego bajtu
        HAL_UART_Receive_IT(&huart1, (uint8_t*)&rx_byte, 1);
    }
}

void HAL_UART_ErrorCallback(UART_HandleTypeDef *huart)
{
    if (huart->Instance == USART1)
    {
        // Ręczne wyczyszczenie błędów
        __HAL_UART_CLEAR_OREFLAG(huart);
        __HAL_UART_CLEAR_NEFLAG(huart);
        __HAL_UART_CLEAR_FEFLAG(huart);
        __HAL_UART_CLEAR_PEFLAG(huart);
        
        // Reset automatu stanów
        rx_state = 0;
        rx_index = 0;
        
        HAL_UART_Receive_IT(&huart1, (uint8_t*)&rx_byte, 1);
    }
}

void UART_SendResponse(UART_Frame_t *frame)
{
    frame->start_byte = UART_FRAME_START_BYTE;
    frame->checksum = UART_CalculateChecksum(frame);
    
    uint8_t tx_buf[5];
    tx_buf[0] = frame->start_byte;
    tx_buf[1] = frame->cmd_type;
    tx_buf[2] = frame->channel;
    tx_buf[3] = frame->data;
    tx_buf[4] = frame->checksum;
    
    // Z uwagi na ogromną prędkość (115200 bps = 5 bajtów zajmuje ~430us) 
    // i rzadkie nadawanie używamy funkcji blokującej
    HAL_UART_Transmit(&huart1, tx_buf, 5, 10);
}
