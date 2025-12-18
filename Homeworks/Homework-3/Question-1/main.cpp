#include "mbed.h"
#include "lib_serial.h"
#include "lib_rng.h"
#include "linear_reg_inference.h"

// Configuration
#define INPUT_PC        1
#define INPUT_MCU       2
#define INPUT           INPUT_PC

float input_buffer[NUM_FEATURES];
float prediction[1];

int main()
{
    // 1. Initialize Serial and RNG
    LIB_UART_Init(); 
    LIB_RNG_Init();

    while (1) {
        #if (INPUT == INPUT_PC)
            // 2a. Receive 5 float features from Python (setup_reg_lr.py)
            if (LIB_SERIAL_Receive(input_buffer, NUM_FEATURES, TYPE_F32) == SERIAL_OK) {
                
                // 3. Perform the calculation
                linear_reg_predict(input_buffer, prediction);

                // 4. Send the result back to Python for verification
                LIB_SERIAL_Transmit(prediction, 1, TYPE_F32);
            }
        #elif (INPUT == INPUT_MCU)
            // 2b. Standalone testing using Random Data
            for (int i = 0; i < NUM_FEATURES; i++) {
                input_buffer[i] = (float)(LIB_RNG_GetRandomNumber() % 1000) / 10.0f;
            }
            // Send input to PC for logging
            LIB_SERIAL_Transmit(input_buffer, NUM_FEATURES, TYPE_F32);
            
            linear_reg_predict(input_buffer, prediction);
            
            // Send output to PC
            LIB_SERIAL_Transmit(prediction, 1, TYPE_F32);
            
            ThisThread::sleep_for(1s);
        #endif
    }
}