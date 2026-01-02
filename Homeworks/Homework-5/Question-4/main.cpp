/*
 * main.cpp
 * FIX: Increased Arena Size to 60KB to prevent memory corruption
 */
#include "mbed.h"
#include "lib_model.h"
#include "lib_serial.h"
#include "temperature_pred_mlp.h"
#include <math.h>

// ---------------------------------------------------------
// 1. HARDWARE
// ---------------------------------------------------------
UnbufferedSerial pc(USBTX, USBRX);
DigitalOut status_led(LED1); 

// ---------------------------------------------------------
// 2. HELPER: SYNCED RECEIVE
// ---------------------------------------------------------
void Serial_ReceiveFrame(uint8_t* buffer, int length) {
    uint8_t val = 0;
    
    // 1. Wait for Sync Byte '$'
    while (val != '$') {
        if (pc.readable()) {
            pc.read(&val, 1);
        }
    }

    // 2. Read Payload
    int bytes_received = 0;
    while (bytes_received < length) {
        if (pc.readable()) {
            pc.read(&buffer[bytes_received], 1);
            bytes_received++;
        }
    }
}

extern "C" void LIB_SERIAL_Init(void) {
    pc.baud(115200);
}

// ---------------------------------------------------------
// 3. CONFIGURATION
// ---------------------------------------------------------
#define NUM_INPUTS 5
#define TYPE_F32 (SERIAL_DataTypeDef)7 

const float TRAIN_MEAN = 18.8249f; 
const float TRAIN_STD = 2.8212f;   

// --- CRITICAL FIX: Increased Arena Size ---
// Was 15 * 1024 (15KB) -> Now 60 * 1024 (60KB)
// This prevents memory overwrites during inference.
constexpr int kTensorArenaSize = 60 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static float input_buffer[NUM_INPUTS]; 
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

// ---------------------------------------------------------
// 4. MAIN LOOP
// ---------------------------------------------------------
int main() {
    // A. BOOT UP
    status_led = 1; wait_us(500 * 1000); status_led = 0;
    LIB_SERIAL_Init();

    // B. HANDSHAKE
    char cmd = 0;
    while (cmd != 'G') {
        pc.write("READY", 5);
        if (pc.readable()) {
            pc.read(&cmd, 1);
        }
        wait_us(500 * 1000);
        status_led = !status_led;
    }
    status_led = 0; 

    // C. INIT MODEL
    // Note: If 60KB is too large for your specific board's RAM, 
    // try reducing to 40 * 1024.
    if (LIB_MODEL_Init(converted_model_tflite, &input, tensor_arena, kTensorArenaSize) != 0) {
        // Error: Fast Blink
        while(1) { status_led = !status_led; wait_us(100 * 1000); } 
    }

    // Safety Check: Input pointer valid?
    if (input == nullptr) {
        while(1) { status_led = !status_led; wait_us(1000 * 1000); } // Slow Blink Error
    }

    // D. PROCESS LOOP
    while (true) {
        // 1. Receive Synced Frame (Waits for '$')
        Serial_ReceiveFrame((uint8_t*)input_buffer, NUM_INPUTS * sizeof(float));

        // 2. Normalize & Populate Input
        for (int i = 0; i < NUM_INPUTS; i++) {
            float normalized = (input_buffer[i] - TRAIN_MEAN) / TRAIN_STD;

            if (input->type == kTfLiteFloat32) {
                input->data.f[i] = normalized;
            } 
            else if (input->type == kTfLiteInt8) {
                int32_t val = (normalized / input->params.scale) + input->params.zero_point;
                val = (val < -128) ? -128 : (val > 127 ? 127 : val);
                input->data.int8[i] = (int8_t)val;
            }
            else if (input->type == kTfLiteUInt8) {
                 int32_t val = (normalized / input->params.scale) + input->params.zero_point;
                 val = (val < 0) ? 0 : (val > 255 ? 255 : val);
                 input->data.uint8[i] = (uint8_t)val;
            }
        }

        // 3. Inference
        if (LIB_MODEL_Run(&output) != 0 || output == nullptr) {
            float err = -999.0f;
            LIB_SERIAL_Transmit(&err, 1, TYPE_F32);
            continue;
        }

        // 4. Read Output
        float raw_output = 0.0f;
        
        if (output->type == kTfLiteFloat32) {
            raw_output = output->data.f[0];
        } 
        else if (output->type == kTfLiteInt8) {
            raw_output = (output->data.int8[0] - output->params.zero_point) * output->params.scale;
        }
        else if (output->type == kTfLiteUInt8) {
            raw_output = (output->data.uint8[0] - output->params.zero_point) * output->params.scale;
        }

        // 5. Final Calculation
        float predicted = (raw_output * TRAIN_STD) + TRAIN_MEAN;
        LIB_SERIAL_Transmit(&predicted, 1, TYPE_F32);
        
        status_led = 1; wait_us(50 * 1000); status_led = 0;
    }
}