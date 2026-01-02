/*
 * main.cpp
 * Question 3: MNIST for BARE METAL (No RTOS)
 */
#include "mbed.h"
#include "lib_model.h"
#include "hdr_mlp.h"
#include "lib_serial.h"
#include "lib_image.h"
#include "hdr_feature_extraction.h"
#include <math.h>

// ---------------------------------------------------------
// 1. HARDWARE SETUP
// ---------------------------------------------------------
UnbufferedSerial pc(USBTX, USBRX);
DigitalOut status_led(LED1); 

// Redirect printf to Serial
FileHandle *mbed::mbed_override_console(int fd) {
    return &pc;
}

// ---------------------------------------------------------
// 2. EXTERNAL FUNCTIONS
// ---------------------------------------------------------
extern "C" void LIB_SERIAL_Init(void) {
    pc.baud(115200);
}

// ---------------------------------------------------------
// 3. MEMORY ALLOCATION
// ---------------------------------------------------------
#define IMG_W 28
#define IMG_H 28
#define NUM_PIXELS (IMG_W * IMG_H)
#define NUM_FEATURES 7
#define NUM_CLASSES 10

#define TYPE_U8  (SERIAL_DataTypeDef)1
#define TYPE_F32 (SERIAL_DataTypeDef)7 

// 30KB Arena (Safe for Bare Metal)
constexpr int kTensorArenaSize = 30 * 1024;
alignas(16) static uint8_t tensor_arena[kTensorArenaSize];

static uint8_t image_buffer[NUM_PIXELS];   
static float feature_buffer[NUM_FEATURES]; 
static TfLiteTensor* input = nullptr;
static TfLiteTensor* output = nullptr;

// ---------------------------------------------------------
// 4. MAIN LOOP
// ---------------------------------------------------------
int main() {
    // A. Boot Signal
    status_led = 1;
    wait_us(1000 * 1000); // 1 Second Delay (Bare Metal compatible)
    status_led = 0;

    // B. Initialize
    LIB_SERIAL_Init();
    printf("\r\n--- STM32 MNIST (Bare Metal) Started ---\r\n");

    if (LIB_MODEL_Init(converted_model_tflite, &input, tensor_arena, kTensorArenaSize) != 0) {
        printf("Error: Model Init Failed\r\n");
        while(1) { 
            status_led = !status_led; 
            wait_us(100 * 1000); // 100ms
        }
    }
    printf("Model Ready.\r\n");

    // C. Ready Signal (3 Slow Blinks)
    for(int i=0; i<6; i++) {
        status_led = !status_led;
        wait_us(300 * 1000); // 300ms
    }
    status_led = 0; 

    while (true) {
        // 1. Receive Image (Blocks here)
        LIB_SERIAL_Receive(image_buffer, NUM_PIXELS, TYPE_U8);

        // 2. Prepare Image Struct
        IMAGE_HandleTypeDef hImg;
        hImg.pData  = image_buffer; 
        hImg.width  = IMG_W;
        hImg.height = IMG_H;
        hImg.format = IMAGE_FORMAT_GRAYSCALE;
        hImg.size   = IMG_W * IMG_H;

        // 3. Extract Features
        HDR_FtrExtOutput feats;
        hdr_calculate_moments(&hImg, &feats);
        hdr_calculate_hu_moments(&feats);
        
        // 4. Log-Scale & Fill Input
        for(int i=0; i<NUM_FEATURES; i++) {
            float val = feats.hu_moments[i];
            if (val == 0.0f) val = 1e-10f; 
            
            float sign = (val > 0) ? 1.0f : -1.0f;
            feature_buffer[i] = -1.0f * sign * log10f(fabsf(val));
            
            input->data.f[i] = feature_buffer[i];
        }

        // 5. Run Model
        if (LIB_MODEL_Run(&output) != 0) {
            float zeros[10] = {0};
            LIB_SERIAL_Transmit(zeros, NUM_CLASSES, TYPE_F32);
            continue;
        }

        // 6. Send Results
        LIB_SERIAL_Transmit(output->data.f, NUM_CLASSES, TYPE_F32);
        
        // Blink Success
        status_led = 1; 
        wait_us(50 * 1000); 
        status_led = 0;
    }
}