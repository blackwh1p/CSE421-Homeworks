import numpy as np
import pandas as pd
from scipy import stats

def create_features(df, time_steps, step_size):
    x_segments = []
    y_segments = []
    z_segments = []
    labels = []
    
    # Iterate through the dataframe to create segments
    for i in range(0, len(df) - time_steps, step_size):
        xs = df["x-accel"].values[i : i + time_steps]
        ys = df["y-accel"].values[i : i + time_steps]
        zs = df["z-accel"].values[i : i + time_steps]

        # Use the label that appears most frequently in this segment
        label = df["activity"].iloc[i : i + time_steps].mode()[0]
        
        # Optional: Ensure segment belongs to one user/activity
        # For simplicity, we assume the mode label represents the segment
        x_segments.append(xs)
        y_segments.append(ys)
        z_segments.append(zs)
        labels.append(label)

    # Reshape segments
    segments_df = pd.DataFrame({
        "x_segments": x_segments,
        "y_segments": y_segments,
        "z_segments": z_segments
    })
    
    feature_df = pd.DataFrame()
    
    # --- Time Domain Features ---
    # 1. Mean
    feature_df['x_mean'] = segments_df["x_segments"].apply(lambda x: x.mean())
    feature_df['y_mean'] = segments_df["y_segments"].apply(lambda x: x.mean())
    feature_df['z_mean'] = segments_df["z_segments"].apply(lambda x: x.mean())

    # 2. Positive Count
    feature_df['x_pos_count'] = segments_df["x_segments"].apply(lambda x: np.sum(x > 0))
    feature_df['y_pos_count'] = segments_df["y_segments"].apply(lambda x: np.sum(x > 0))
    feature_df['z_pos_count'] = segments_df["z_segments"].apply(lambda x: np.sum(x > 0))

    # --- Frequency Domain Features (FFT) ---
    FFT_SIZE = time_steps // 2 + 1
    
    x_fft_series = segments_df["x_segments"].apply(lambda x: np.abs(np.fft.fft(x))[1:FFT_SIZE])
    y_fft_series = segments_df["y_segments"].apply(lambda x: np.abs(np.fft.fft(x))[1:FFT_SIZE])
    z_fft_series = segments_df["z_segments"].apply(lambda x: np.abs(np.fft.fft(x))[1:FFT_SIZE])

    # 3. FFT Standard Deviation
    feature_df['x_std_fft'] = x_fft_series.apply(lambda x: x.std())
    feature_df['y_std_fft'] = y_fft_series.apply(lambda x: x.std())
    feature_df['z_std_fft'] = z_fft_series.apply(lambda x: x.std())

    # 4. FFT Signal Magnitude Area (SMA)
    # Book logic: Sum of absolute values divided by 50 (normalization factor)
    feature_df['sma_fft'] = x_fft_series.apply(lambda x: np.sum(np.abs(x)/50)) + \
                            y_fft_series.apply(lambda x: np.sum(np.abs(x)/50)) + \
                            z_fft_series.apply(lambda x: np.sum(np.abs(x)/50))

    labels = np.asarray(labels)

    return feature_df, labels