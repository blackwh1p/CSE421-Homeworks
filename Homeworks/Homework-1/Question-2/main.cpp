#include "mbed.h"
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include "mfcc.h"

// CONFIG
static const uint32_t FRAME_LEN = 1024;   // must match host
static const uint32_t NUM_MELS  = 20;
static const uint32_t NUM_DCT   = 13;
static const int      BAUD      = 115200;

// ST-LINK VCP
static UnbufferedSerial pc(USBTX, USBRX, BAUD);

static bool read_exact(uint8_t* dst, size_t n) {
    size_t got = 0;
    while (got < n) {
        ssize_t r = pc.read(dst + got, n - got);
        if (r <= 0) { ThisThread::sleep_for(1ms); continue; }
        got += (size_t)r;
    }
    return true;
}

int main() {
    pc.set_blocking(true);
    printf("\r\n[STM32 MFCC] Ready. Send frames with host.\r\n");

    mfcc_instance_t S;
    if (mfcc_init(&S, FRAME_LEN, NUM_MELS, NUM_DCT) != 0) {
        printf("MFCC init failed\r\n");
        while (true) { ThisThread::sleep_for(1s); }
    }

    int16_t* frame = (int16_t*)malloc(FRAME_LEN * sizeof(int16_t));
    float*   mfcc  = (float*)malloc(NUM_DCT * sizeof(float));
    if (!frame || !mfcc) {
        printf("malloc failed\r\n");
        while (true) { ThisThread::sleep_for(1s); }
    }

    while (true) {
        // 1) header 'W'
        uint8_t hdr = 0;
        if (!read_exact(&hdr, 1)) continue;
        if (hdr != 'W') continue;

        // 2) uint16 length (little-endian)
        uint8_t lenBytes[2];
        if (!read_exact(lenBytes, 2)) continue;
        uint16_t n = (uint16_t)(lenBytes[0] | (lenBytes[1] << 8));

        if (n != FRAME_LEN) {
            // drain unexpected payload
            size_t toDrain = (size_t)n * sizeof(int16_t);
            uint8_t dump[64];
            while (toDrain) {
                size_t chunk = toDrain > sizeof(dump) ? sizeof(dump) : toDrain;
                read_exact(dump, chunk);
                toDrain -= chunk;
            }
            printf("Bad frame size %u (expected %u)\r\n", n, FRAME_LEN);
            continue;
        }

        // 3) payload: int16 PCM
        if (!read_exact((uint8_t*)frame, FRAME_LEN * sizeof(int16_t))) {
            printf("Read error\r\n");
            continue;
        }

        // 4) MFCC
        mfcc_compute(&S, frame, mfcc);

        // 5) print CSV: c0..c12
        for (uint32_t i = 0; i < NUM_DCT; i++) {
            printf("%s%.6f", (i ? "," : ""), mfcc[i]);
        }
        printf("\r\n");
    }
}