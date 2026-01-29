#ifndef SPI_COMM_H
#define SPI_COMM_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "stm32f0xx_hal.h"

#define SPI_BUFFER_SIZE 64
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
        SPI_CMD_ACK = 0xAA,       // Potwierdzenie
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

    // Stan parsera ramek
    typedef enum
    {
        SPI_STATE_IDLE,
        SPI_STATE_WAIT_CMD,
        SPI_STATE_WAIT_CHANNEL,
        SPI_STATE_WAIT_DATA,
        SPI_STATE_WAIT_CHECKSUM,
        SPI_STATE_PROCESS
    } SPI_ParserState;

    // Eksport globalnych zmiennych SPI
    extern SPI_HandleTypeDef hspi1;
    extern volatile uint8_t spi_rx_buffer[SPI_BUFFER_SIZE];
    extern volatile uint8_t spi_tx_buffer[SPI_BUFFER_SIZE];
    extern volatile uint8_t spi_rx_index;
    extern volatile uint8_t spi_data_ready;

    // Parser ramek
    extern SPI_Frame_t current_frame;
    extern SPI_ParserState parser_state;
    extern volatile uint8_t frame_ready;

    // Funkcje SPI
    void SPI_Init(void);
    void SPI_SendData(uint8_t *data, uint16_t size);
    void SPI_SendFrame(SPI_Frame_t *frame);
    uint8_t SPI_GetLastReceivedByte(void);
    void SPI_ProcessByte(uint8_t byte);
    uint8_t SPI_CalculateChecksum(SPI_Frame_t *frame);
    uint8_t SPI_IsFrameValid(SPI_Frame_t *frame);

#ifdef __cplusplus
}
#endif

#endif // SPI_COMM_H
