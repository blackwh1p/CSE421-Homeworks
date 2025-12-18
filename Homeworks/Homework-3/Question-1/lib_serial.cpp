#include "lib_serial.h"

// Initialize the Serial port over the USB cable
// USBTX and USBRX are the internal pins for the ST-Link
static BufferedSerial pc(USBTX, USBRX, 115200);

void LIB_UART_Init(void) {
    pc.set_format(8, BufferedSerial::None, 1);
}

int8_t LIB_SERIAL_Transmit(void *pData, uint32_t length, SERIAL_DataTypeDef type) {
    uint8_t header[3] = {'S', 'T', 'W'};
    uint32_t byte_length = 0;
    uint8_t *ptr = (uint8_t*) pData;

    if (type <= 2) byte_length = length;
    else if (type <= 4) byte_length = length * 2;
    else byte_length = length * 4;

    pc.write(header, 3);
    pc.write(&type, 1);
    pc.write(&byte_length, 4);
    pc.write(ptr, byte_length);
    
    return SERIAL_OK;
}

int8_t LIB_SERIAL_Receive(void *pData, uint32_t length, SERIAL_DataTypeDef type) {
    uint8_t header[3] = {'S', 'T', 'R'};
    uint32_t byte_length = 0;
    uint8_t *ptr = (uint8_t*) pData;

    if (type <= 2) byte_length = length;
    else if (type <= 4) byte_length = length * 2;
    else byte_length = length * 4;

    pc.write(header, 3);
    pc.write(&type, 1);
    pc.write(&byte_length, 4);

    uint32_t total = 0;
    while (total < byte_length) {
        if (pc.readable()) {
            total += pc.read(ptr + total, byte_length - total);
        }
    }
    return SERIAL_OK;
}