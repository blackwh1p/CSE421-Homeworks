/*
 * lib_serial.cpp
 * Mbed Version - MODIFIED for Raw Data (No "STW" Headers)
 */
#include "lib_serial.h"
#include "mbed.h"

// External reference to the Serial object defined in main.cpp
extern UnbufferedSerial pc; 

/**
 * @brief Transmits data via Serial WITHOUT headers (Raw Binary)
 * @param pData   Pointer to data buffer
 * @param length  Number of elements (not bytes!)
 * @param type    Data type (to calculate total bytes)
 */
int8_t LIB_SERIAL_Transmit(void *pData, uint32_t length, SERIAL_DataTypeDef type)
{
    uint32_t total_bytes = 0;
    uint8_t *byteData = (uint8_t*)pData;

    // 1. Calculate total bytes based on type
    if ((type == TYPE_S32) || (type == TYPE_U32) || (type == TYPE_F32))
    {
        total_bytes = length * 4;
    }
    else if ((type == TYPE_S16) || (type == TYPE_U16))
    {
        total_bytes = length * 2;
    }
    else
    {
        total_bytes = length; // 1 byte types
    }

    // 2. Send Raw Data Only (No Headers)
    // This allows your Python script to read the data directly 
    // without getting confused by "STW" characters.
    if (pc.write(byteData, total_bytes) != total_bytes)
    {
        return SERIAL_ERROR;
    }

    return SERIAL_OK;
}

/**
 * @brief Placeholder for Receive (Not strictly used in this KWS implementation)
 */
int8_t LIB_SERIAL_Receive(void *pData, uint32_t length, SERIAL_DataTypeDef type)
{
    // Not implemented for this simplified version
    return SERIAL_ERROR;
}