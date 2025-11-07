#ifndef MFCC_H
#define MFCC_H

#include <stdint.h>
#include "arm_math.h"

#ifdef __cplusplus
extern "C" {
#endif

// ===== Config (match main.cpp & host) =====
#ifndef SAMP_FREQ
#define SAMP_FREQ 8000U
#endif

#ifndef FFT_MAX
#define FFT_MAX 1024u
#endif

#ifndef NMEL_MAX
#define NMEL_MAX 32u     // supports up to 32 mel filters
#endif

#ifndef NDCT_MAX
#define NDCT_MAX 13u     // supports up to 13 MFCC outputs
#endif

// Worst-case packed weights: each mel can cover ~ (FFT/2+1) bins
#ifndef PACK_MAX
#define PACK_MAX (NMEL_MAX * ((FFT_MAX/2u)+1u))
#endif

typedef struct {
    uint32_t fftLen;           // <= FFT_MAX (power of two)
    uint32_t nbMelFilters;     // <= NMEL_MAX
    uint32_t nbDctOutputs;     // <= NDCT_MAX

    // pointers mapped to static buffers (no malloc)
    float    *windowCoefs;     // [fftLen]
    float    *dctMatrix;       // [nbDctOutputs * nbMelFilters]

    uint32_t *filterPos;       // [nbMelFilters]
    uint32_t *filterLen;       // [nbMelFilters]
    uint32_t *filterOffset;    // [nbMelFilters]
    float    *packedFilters;   // <= PACK_MAX

    arm_rfft_fast_instance_f32 rfft;
} mfcc_instance_t;

int  mfcc_init  (mfcc_instance_t *S, uint32_t fftLen, uint32_t nMels, uint32_t nDct);
void mfcc_free  (mfcc_instance_t *S); // no-op (kept for API symmetry)
void mfcc_compute(const mfcc_instance_t *S, const int16_t *audio, float *mfcc_out);

#ifdef __cplusplus
}
#endif
#endif