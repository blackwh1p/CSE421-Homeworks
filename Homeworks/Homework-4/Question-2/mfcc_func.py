import os
import numpy as np
from scipy.io import wavfile
import cmsisdsp as dsp
import cmsisdsp.mfcc as mfcc
from cmsisdsp.datatype import F32

def create_mfcc_features(recordings_list, FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs, window):
    # Initialize CMSIS-DSP MFCC instance
    # We use F32 (Float 32) datatype
    freq_min = 20
    freq_high = sample_rate / 2
    
    # Generate filter matrix
    filtLen, filtPos, packedFilters = mfcc.melFilterMatrix(
        F32, freq_min, freq_high, numOfMelFilters, sample_rate, FFTSize
    )

    # Generate DCT matrix
    dctMatrixFilters = mfcc.dctMatrix(F32, numOfDctOutputs, numOfMelFilters)

    num_samples = len(recordings_list)
    # The output size is (numOfDctOutputs * 2) because we stack 2 frames (first half + second half)
    mfcc_features = np.empty((num_samples, numOfDctOutputs * 2), dtype=np.float32)
    labels = np.empty(num_samples)

    mfccf32 = dsp.arm_mfcc_instance_f32()

    # Initialize the ARM MFCC instance
    status = dsp.arm_mfcc_init_f32(
        mfccf32,
        FFTSize,
        numOfMelFilters,
        numOfDctOutputs,
        dctMatrixFilters,
        filtPos,
        filtLen,
        packedFilters,
        window,
    )

    print(f"Extracting features from {num_samples} audio files...")

    for sample_idx, wav_path in enumerate(recordings_list):
        # Parse filename to get label (e.g., "0_jackson_0.wav" -> digit is 0)
        wav_file = os.path.basename(wav_path)
        file_specs = wav_file.split(".")[0]
        parts = file_specs.split("_")
        
        if len(parts) < 1:
            print(f"Skipping malformed file: {wav_file}")
            continue
            
        digit = int(parts[0])

        # Read Audio
        try:
            _, sample = wavfile.read(wav_path)
        except ValueError:
            print(f"Error reading {wav_path}")
            continue
            
        # Ensure float32
        sample = sample.astype(np.float32)
        
        # Take the first 2 windows worth of data (2 * FFTSize)
        # If file is too short, pad it. If too long, crop it.
        limit = 2 * FFTSize
        sample = sample[:limit]
        
        # FIX: The original code had a syntax error here: len(sample < ...)
        if len(sample) < limit:
            padding_needed = limit - len(sample)
            sample = np.pad(sample, (0, padding_needed), "constant", constant_values=0)
            
        # Normalize audio volume
        max_val = max(abs(sample))
        if max_val > 0:
            sample = sample / max_val
            
        # Split into two frames
        first_half = sample[:FFTSize]
        second_half = sample[FFTSize : 2 * FFTSize]
        
        # CMSIS-DSP MFCC requires a temp buffer of size FFTSize + 2
        tmp = np.zeros(FFTSize + 2, dtype=np.float32)
        
        # Compute MFCC for both frames
        first_half_mfcc = dsp.arm_mfcc_f32(mfccf32, first_half, tmp)
        second_half_mfcc = dsp.arm_mfcc_f32(mfccf32, second_half, tmp)
        
        # Concatenate results
        mfcc_feature = np.concatenate((first_half_mfcc, second_half_mfcc))
        
        # Store
        mfcc_features[sample_idx] = mfcc_feature
        labels[sample_idx] = digit
        
    return mfcc_features, labels