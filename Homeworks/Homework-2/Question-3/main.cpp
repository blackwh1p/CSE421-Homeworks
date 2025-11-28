#include "mbed.h"
#include "arm_math.h"
#include "lib_image.h"
#include "hdr_feature_extraction.h"
#include "dt_cls_inference.h" // Use DT header

#define BAUD_RATE 115200
#define IMG_SIZE 784 // 28x28

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
        // 1. Sync
        char ready = 'R';
        pc.write(&ready, 1);

        // 2. Receive Image
        char* ptr = (char*)img_buffer;
        int remaining = IMG_SIZE;
        while (remaining > 0) {
            if (pc.readable()) {
                int n = pc.read(ptr, remaining);
                remaining -= n;
                ptr += n;
            }
        }

        // 3. Extract Features
        // Because we transposed in Python, these features will now match!
        hdr_calculate_moments(&img_handle, &features);
        hdr_calculate_hu_moments(&features);

        // 4. Map features to array
        for(int i=0; i<7; i++) {
            hu_moments_array[i] = features.hu_moments[i];
        }

        // 5. Inference (Decision Tree)
        // Fills 'class_votes' with predictions
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