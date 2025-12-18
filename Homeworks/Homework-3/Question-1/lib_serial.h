#ifndef INC_LIB_SERIAL_H_
#define INC_LIB_SERIAL_H_

#include "mbed.h"

typedef enum {
    TYPE_U8  = 1,
    TYPE_S8  = 2,
    TYPE_U16 = 3,
    TYPE_S16 = 4,
    TYPE_U32 = 5,
    TYPE_S32 = 6,
    TYPE_F32 = 7,
} SERIAL_DataTypeDef;

#define SERIAL_OK      (0)
#define SERIAL_ERROR   (-1)

// --- ENSURE THIS LINE IS PRESENT ---
void LIB_UART_Init(void); 

int8_t LIB_SERIAL_Transmit(void *pData, uint32_t length, SERIAL_DataTypeDef type);
int8_t LIB_SERIAL_Receive(void *pData, uint32_t length, SERIAL_DataTypeDef type);

#endif