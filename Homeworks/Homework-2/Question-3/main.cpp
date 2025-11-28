#include "mbed.h"
// Ensure you include arm_math.h if your library relies on it, 
// otherwise standard math.h is fine for the corrected feature extraction
#include "lib_image.h"
#include "hdr_feature_extraction.h"
#include "dt_cls_inference.h"

#define BAUD_RATE 115200
#define IMG_SIZE 784 // 28x28

// Use the specific pins for your board (DISCO-F746NG uses USBTX/USBRX)
UnbufferedSerial pc(USBTX, USBRX);
uint8_t img_buffer[IMG_SIZE];

// Objects
IMAGE_HandleTypeDef img_handle;
HDR_FtrExtOutput features;

// Feature array for DT
float hu_moments_array[7];
// DT returns votes/probabilities (array of size 10)
int class_votes[10]; 

int main() {
    pc.baud(BAUD_RATE);

    // Init Image Struct
    img_handle.pData = img_buffer;
    img_handle.width = 28;
    img_handle.height = 28;
    img_handle.format = IMAGE_FORMAT_GRAYSCALE;
    img_handle.size = IMG_SIZE;

    while (true) {
        // 1. Sync - Send 'R' to tell Python we are ready
        char ready = 'R';
        if (pc.writable()) {
            pc.write(&ready, 1);
        }

        // 2. Receive Image (FIXED LOGIC)
        char* ptr = (char*)img_buffer;
        int remaining = IMG_SIZE;
        
        while (remaining > 0) {
            if (pc.readable()) {
                // Use ssize_t to capture error codes
                ssize_t n = pc.read(ptr, remaining);
                
                // Only update pointers if data was actually read (n > 0)
                if (n > 0) {
                    remaining -= n;
                    ptr += n;
                }
            }
        }

        // 3. Extract Features
        hdr_calculate_moments(&img_handle, &features);
        hdr_calculate_hu_moments(&features);

        // 4. Map features to array
        for(int i=0; i<7; i++) {
            hu_moments_array[i] = features.hu_moments[i];
        }

        // 5. Inference (Decision Tree)
        // Zero out votes before prediction just in case
        memset(class_votes, 0, sizeof(class_votes));
        dt_cls_predict(hu_moments_array, class_votes);

        // 6. Argmax (Find predicted class)
        int best_class = 0;
        int max_vote = -1;
        
        for(int i=0; i<10; i++) {
            if (class_votes[i] > max_vote) {
                max_vote = class_votes[i];
                best_class = i;
            }
        }

        // 7. Send Result
        char res = (char)best_class;
        pc.write(&res, 1);
    }
}