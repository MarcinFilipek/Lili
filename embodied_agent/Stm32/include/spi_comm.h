#ifndef SPI_COMM_H
#define SPI_COMM_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "stm32f0xx_hal.h"

#define SPI_FRAME_START_BYTE 0xAA

    // Typy komend
    typedef enum
    {
        SPI_CMD_SET_SERVO = 0x01, // Ustaw kąt serwa
        SPI_CMD_GET_SERVO = 0x02, // Odczytaj kąt serwa
        SPI_CMD_SET_PWM = 0x03,   // Ustaw PWM
        SPI_CMD_GET_PWM = 0x04,   // Odczytaj PWM
        SPI_CMD_PING = 0x05,      // Test połączenia
        SPI_CMD_RESET = 0x06,     // Reset animacji
        SPI_CMD_ACK = 0xCC,       // Potwierdzenie
        SPI_CMD_NACK = 0xEE       // Błąd
    } SPI_CommandType;

    // Struktura ramki SPI
    typedef struct
    {
        uint8_t start_byte; // 0xAA - znacznik początku
        uint8_t cmd_type;   // Typ komendy
        uint8_t channel;    // Numer kanału (0-15)
        uint8_t data;       // Dane (kąt, PWM, itp.)
        uint8_t checksum;   // Suma kontrolna
    } SPI_Frame_t;

    // Rozmiar kolejki komend (musi być potęgą 2)
#define SPI_CMD_QUEUE_SIZE 8

    // Ring buffer komend
    typedef struct
    {
        SPI_Frame_t frames[SPI_CMD_QUEUE_SIZE];
        volatile uint8_t head; // zapisuje ISR
        volatile uint8_t tail; // czyta main loop
    } SPI_CmdQueue_t;

    // Eksport globalnych zmiennych SPI
    extern SPI_HandleTypeDef hspi1;
    extern volatile uint8_t spi_rx_frame[5];
    extern volatile uint8_t spi_tx_frame[5];

    extern volatile uint8_t response_pending;

    // Kolejka komend
    extern SPI_CmdQueue_t spi_cmd_queue;

    // Funkcje SPI
    void SPI_Init(void);
    void SPI_PrepareResponse(SPI_Frame_t *frame);
    uint8_t SPI_CalculateChecksum(SPI_Frame_t *frame);
    uint8_t SPI_IsFrameValid(SPI_Frame_t *frame);
    uint8_t SPI_Queue_Pop(SPI_Frame_t *out_frame);

#ifdef __cplusplus
}
#endif

#endif // SPI_COMM_H
