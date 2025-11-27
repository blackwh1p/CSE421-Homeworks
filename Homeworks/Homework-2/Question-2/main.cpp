#include "mbed.h"

extern "C" {
#include "knn_cls_inference.h"   // provides knn_cls_predict(), NUM_CLASSES
#include "ks_feature_extraction.h" // only for nbDctOutputs definition
}

// Serial port to PC (ST-LINK VCP on most Nucleo/Disco boards)
BufferedSerial pc(USBTX, USBRX, 115200);   // same baud as old code (2 Mbps)

// Features: same size as before (nbDctOutputs * 2)
float32_t ExtractedFeatures[nbDctOutputs * 2] = {0};
int32_t   output[NUM_CLASSES] = {0};

int main()
{
    pc.set_blocking(true);

    const uint32_t num_features = nbDctOutputs * 2;
    const uint32_t bytes_to_read = num_features * sizeof(float32_t);
    const uint32_t bytes_to_write = NUM_CLASSES * sizeof(int32_t);

    while (true) {
        uint8_t *rx_ptr = reinterpret_cast<uint8_t*>(ExtractedFeatures);
        uint32_t received = 0;

        // ---- Receive feature vector from PC (float32 LE) ----
        while (received < bytes_to_read) {
            ssize_t n = pc.read(rx_ptr + received, bytes_to_read - received);
            if (n > 0) {
                received += static_cast<uint32_t>(n);
            }
        }

        // ---- Run KNN classifier on the received features ----
        knn_cls_predict(ExtractedFeatures, output);

        // ---- Send prediction vector back to PC (int32) ----
        uint8_t *tx_ptr = reinterpret_cast<uint8_t*>(output);
        uint32_t sent = 0;
        while (sent < bytes_to_write) {
            ssize_t n = pc.write(tx_ptr + sent, bytes_to_write - sent);
            if (n > 0) {
                sent += static_cast<uint32_t>(n);
            }
        }
    }
}
