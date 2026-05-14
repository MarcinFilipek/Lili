#ifndef UART_COMM_H
#define UART_COMM_H

#include "stm32f0xx_hal.h"

// Protokół komunikacyjny (format ramki 5-bajtowej)
#define UART_FRAME_START_BYTE 0xAA

#define UART_CMD_SET_SERVO    0x01
#define UART_CMD_GET_SERVO    0x02
#define UART_CMD_SET_PWM      0x03
#define UART_CMD_PING         0x04
#define UART_CMD_RESET        0x05
#define UART_CMD_ACK          0xCC
#define UART_CMD_NACK         0xFF

// Struktura ramki UART (5 bajtów)
typedef struct {
    uint8_t start_byte;
    uint8_t cmd_type;
    uint8_t channel;
    uint8_t data;
    uint8_t checksum;
} UART_Frame_t;

// Rozmiar bufora komend (musi być potęgą 2)
#define UART_CMD_QUEUE_SIZE 16

typedef struct {
    UART_Frame_t frames[UART_CMD_QUEUE_SIZE];
    volatile uint8_t head;
    volatile uint8_t tail;
} UART_CmdQueue_t;

// Deklaracje funkcji
void UART_Init(void);
uint8_t UART_Queue_Pop(UART_Frame_t *out_frame);
void UART_SendResponse(UART_Frame_t *frame);
uint8_t UART_CalculateChecksum(UART_Frame_t *frame);

extern UART_HandleTypeDef huart1;

#endif // UART_COMM_H
