#ifndef LIB_SERIAL_H
#define LIB_SERIAL_H

#include "mbed.h"

typedef enum {
    TYPE_U8 = 1, TYPE_S8 = 2, TYPE_U16 = 3, TYPE_S16 = 4,
    TYPE_U32 = 5, TYPE_S32 = 6, TYPE_F32 = 7
} SERIAL_DataTypeDef;

// UPDATED: Now accepts PinName arguments
void LIB_SERIAL_Init(PinName tx, PinName rx);
void LIB_SERIAL_Transmit(void *pData, uint32_t length, SERIAL_DataTypeDef type);
void LIB_SERIAL_Receive(void *pData, uint32_t length, SERIAL_DataTypeDef type);

#endif