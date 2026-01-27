#ifndef SPI_COMM_H
#define SPI_COMM_H

#ifdef __cplusplus
extern "C"
{
#endif

#include "stm32f0xx_hal.h"

#define SPI_BUFFER_SIZE 64

    // Eksport globalnych zmiennych SPI
    extern SPI_HandleTypeDef hspi1;
    extern volatile uint8_t spi_rx_buffer[SPI_BUFFER_SIZE];
    extern volatile uint8_t spi_tx_buffer[SPI_BUFFER_SIZE];
    extern volatile uint8_t spi_rx_index;
    extern volatile uint8_t spi_data_ready;

    // Funkcje SPI
    void SPI_Init(void);
    void SPI_SendData(uint8_t *data, uint16_t size);
    uint8_t SPI_GetLastReceivedByte(void);

#ifdef __cplusplus
}
#endif

#endif // SPI_COMM_H
