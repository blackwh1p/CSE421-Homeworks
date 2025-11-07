#include "mfcc.h"
#include <string.h>
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// ===== Static buffers (single instance) =====
static float    g_window[FFT_MAX];
static float    g_dct[NDCT_MAX * NMEL_MAX];

static uint32_t g_fPos[NMEL_MAX];
static uint32_t g_fLen[NMEL_MAX];
static uint32_t g_fOffs[NMEL_MAX];
static float    g_pack[PACK_MAX];

static float    g_melC[NMEL_MAX + 2u];

// scratch for compute (reuse each call)
static float    g_frame[FFT_MAX];
static float    g_fftbuf[FFT_MAX];
static float    g_pows[(FFT_MAX/2u) + 1u];
static float    g_melE[NMEL_MAX];

// ===== helpers =====
static inline float freq2mel(float f) { return 1127.0f * logf(1.0f + f / 700.0f); }
static inline float mel2freq(float m) { return 700.0f  * (expf(m / 1127.0f) - 1.0f); }

static void make_hamming(float *w, uint32_t N) {
    if (N < 2) { if (N == 1) w[0] = 1.0f; return; }
    const float two_pi = 2.0f * (float)M_PI;
    for (uint32_t n = 0; n < N; n++) {
        w[n] = 0.54f - 0.46f * cosf(two_pi * (float)n / (float)(N - 1));
    }
}

static void make_dct(float *M, uint32_t nDct, uint32_t nMels) {
    const float norm = sqrtf(2.0f / (float)nMels);
    for (uint32_t k = 0; k < nDct; k++) {
        for (uint32_t n = 0; n < nMels; n++) {
            float s = ((float)n + 0.5f) / (float)nMels;
            M[k * nMels + n] = cosf((float)k * (float)M_PI * s) * norm;
        }
    }
}

static int build_mel_filterbank(uint32_t fftLen, uint32_t nMels,
                                uint32_t *pos, uint32_t *len, uint32_t *offs,
                                float *packed, uint32_t *packed_used)
{
    const uint32_t half   = fftLen / 2u;
    const float    fmin_m = freq2mel(20.0f);
    const float    fmax_m = freq2mel((float)SAMP_FREQ * 0.5f); // Nyquist
    const float    bin_hz = (float)SAMP_FREQ / (float)fftLen;

    // centers on mel scale (nMels+2)
    const float step = (fmax_m - fmin_m) / (float)(nMels + 1u);
    for (uint32_t i = 0; i < nMels + 2u; i++) g_melC[i] = fmin_m + step * (float)i;

    // first pass: spans & total
    uint32_t total = 0;
    for (uint32_t m = 0; m < nMels; m++) {
        float fL = mel2freq(g_melC[m]);
        float fC = mel2freq(g_melC[m+1]);
        float fR = mel2freq(g_melC[m+2]);

        uint32_t bL = (uint32_t)floorf(fL / bin_hz);
        uint32_t bC = (uint32_t)floorf(fC / bin_hz);
        uint32_t bR = (uint32_t)floorf(fR / bin_hz);

        if (bL < 1u)  bL = 1u;         // skip DC bin
        if (bR > half) bR = half;      // clamp to Nyquist
        if (bC <= bL)  bC = bL + 1u;
        if (bR <= bC)  bR = bC + 1u;

        pos[m] = bL;
        len[m] = (bR - bL + 1u);
        total += len[m];
    }
    if (total > PACK_MAX) return -1;

    // second pass: fill packed weights
    uint32_t cur = 0;
    for (uint32_t m = 0; m < nMels; m++) {
        float fL = mel2freq(g_melC[m]);
        float fC = mel2freq(g_melC[m+1]);
        float fR = mel2freq(g_melC[m+2]);

        uint32_t bL = pos[m];
        uint32_t bR = bL + len[m] - 1u;
        uint32_t bC = (uint32_t)floorf(fC / bin_hz);
        if (bC <= bL) bC = bL + 1u;
        if (bC >= bR) bC = bR - 1u;

        offs[m] = cur;

        for (uint32_t b = bL; b <= bC; b++) {
            float hz = (float)b * bin_hz;
            float w  = (hz - fL) / (fC - fL);
            if (w < 0.0f) w = 0.0f; if (w > 1.0f) w = 1.0f;
            packed[cur++] = w;
        }
        for (uint32_t b = bC + 1u; b <= bR; b++) {
            float hz = (float)b * bin_hz;
            float w  = (fR - hz) / (fR - fC);
            if (w < 0.0f) w = 0.0f; if (w > 1.0f) w = 1.0f;
            packed[cur++] = w;
        }
    }
    *packed_used = cur;
    return 0;
}

// ===== API =====
int mfcc_init(mfcc_instance_t *S, uint32_t fftLen, uint32_t nMels, uint32_t nDct) {
    if (!S) return -1;
    if ((fftLen & (fftLen - 1u)) != 0u) return -2; // must be power of 2
    if (fftLen > FFT_MAX || nMels > NMEL_MAX || nDct > NDCT_MAX) return -3;

    memset(S, 0, sizeof(*S));
    S->fftLen       = fftLen;
    S->nbMelFilters = nMels;
    S->nbDctOutputs = nDct;

    // map pointers to static buffers
    S->windowCoefs  = g_window;
    S->dctMatrix    = g_dct;
    S->filterPos    = g_fPos;
    S->filterLen    = g_fLen;
    S->filterOffset = g_fOffs;
    S->packedFilters= g_pack;

    // fill window and DCT
    make_hamming(S->windowCoefs, fftLen);
    make_dct(S->dctMatrix, nDct, nMels);

    // build mel filterbank
    uint32_t used = 0;
    if (build_mel_filterbank(fftLen, nMels, S->filterPos, S->filterLen,
                             S->filterOffset, S->packedFilters, &used) != 0)
        return -4;
    (void)used; // not needed later

    // CMSIS RFFT init
    if (arm_rfft_fast_init_f32(&S->rfft, fftLen) != ARM_MATH_SUCCESS) return -5;

    return 0;
}

void mfcc_free(mfcc_instance_t *S) {
    (void)S; // nothing to free (static buffers)
}

void mfcc_compute(const mfcc_instance_t *S, const int16_t *audio, float *mfcc_out) {
    const uint32_t N  = S->fftLen;
    const uint32_t NH = N / 2u;

    // int16 → float + window
    for (uint32_t i = 0; i < N; i++) {
        g_frame[i] = ((float)audio[i] / 32768.0f) * S->windowCoefs[i];
    }

    // RFFT
    arm_rfft_fast_f32((arm_rfft_fast_instance_f32*)&S->rfft, g_frame, g_fftbuf, 0);

    // power spectrum (rfft layout: re0, reN/2, re1, im1, ...)
    g_pows[0]  = g_fftbuf[0] * g_fftbuf[0];
    g_pows[NH] = g_fftbuf[1] * g_fftbuf[1];
    for (uint32_t k = 1; k < NH; k++) {
        float re = g_fftbuf[2u*k + 0u];
        float im = g_fftbuf[2u*k + 1u];
        g_pows[k] = re*re + im*im;
    }

    // mel energies (log of weighted magnitude)
    for (uint32_t m = 0; m < S->nbMelFilters; m++) {
        uint32_t start = S->filterPos[m];
        uint32_t L     = S->filterLen[m];
        uint32_t off   = S->filterOffset[m];

        float acc = 0.0f;
        for (uint32_t j = 0; j < L; j++) {
            float mag = sqrtf(g_pows[start + j]);
            acc += mag * S->packedFilters[off + j];
        }
        if (acc < 1e-12f) acc = 1e-12f;
        g_melE[m] = logf(acc);
    }

    // DCT → MFCC
    for (uint32_t i = 0; i < S->nbDctOutputs; i++) {
        const float *row = &S->dctMatrix[i * S->nbMelFilters];
        float sum = 0.0f;
        for (uint32_t m = 0; m < S->nbMelFilters; m++) sum += row[m] * g_melE[m];
        mfcc_out[i] = sum;
    }
}