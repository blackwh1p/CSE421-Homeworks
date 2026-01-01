#include "lib_serial.h"

// Pointer to the serial object
static BufferedSerial *serial_port = nullptr;

void LIB_SERIAL_Init(PinName tx, PinName rx) {
    // Create the serial object dynamically with the provided pins
    static BufferedSerial s(tx, rx);
    s.set_baud(115200);
    serial_port = &s;
}

void LIB_SERIAL_Transmit(void *pData, uint32_t length, SERIAL_DataTypeDef type) {
    if (!serial_port) return; // Guard against uninitialized usage

    uint32_t byte_len = length;
    if(type == TYPE_F32 || type == TYPE_U32 || type == TYPE_S32) byte_len *= 4;
    else if(type == TYPE_U16 || type == TYPE_S16) byte_len *= 2;

    serial_port->write(pData, byte_len);
}

void LIB_SERIAL_Receive(void *pData, uint32_t length, SERIAL_DataTypeDef type) {
    if (!serial_port) return;

    uint32_t byte_len = length;
    if(type == TYPE_F32 || type == TYPE_U32 || type == TYPE_S32) byte_len *= 4;
    else if(type == TYPE_U16 || type == TYPE_S16) byte_len *= 2;

    uint8_t *ptr = (uint8_t *)pData;
    uint32_t received = 0;

    while (received < byte_len) {
        if (serial_port->readable()) {
            int n = serial_port->read(ptr + received, byte_len - received);
            if (n > 0) received += n;
        }
    }
}