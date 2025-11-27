import numpy as np
import pandas as pd

def create_features(df, time_steps, step_size):
    x_segments = []
    y_segments = []
    z_segments = []
    labels = []
    
    for i in range(0, len(df) - time_steps, step_size):
        xs = df["x-accel"].values[i : i + time_steps]
        ys = df["y-accel"].values[i : i + time_steps]
        zs = df["z-accel"].values[i : i + time_steps]

        count_per_label = df["activity"][i : i + time_steps].value_counts()
        if not count_per_label.empty:
            label_count = count_per_label.iloc[0]
            if label_count == time_steps:
                labels.append(count_per_label.index[0])
                x_segments.append(xs)
                y_segments.append(ys)
                z_segments.append(zs)

    segments_df = pd.DataFrame({
        "x_segments": x_segments,
        "y_segments": y_segments,
        "z_segments": z_segments
    })
    
    feature_df = pd.DataFrame()
    
    # 1. Mean
    feature_df['x_mean'] = segments_df["x_segments"].apply(lambda x: x.mean())
    feature_df['y_mean'] = segments_df["y_segments"].apply(lambda x: x.mean())
    feature_df['z_mean'] = segments_df["z_segments"].apply(lambda x: x.mean())

    # 2. Positive Count
    feature_df['x_pos_count'] = segments_df["x_segments"].apply(lambda x: np.sum(x > 0))
    feature_df['y_pos_count'] = segments_df["y_segments"].apply(lambda x: np.sum(x > 0))
    feature_df['z_pos_count'] = segments_df["z_segments"].apply(lambda x: np.sum(x > 0))
    
    def get_fft_features(series):
        # 1. Compute RFFT
        fft_vals = np.abs(np.fft.rfft(series))
        # 2. Slice EXACTLY how the C code sees it (Indices 1 to 31)
        # [1:32] means start at 1, stop BEFORE 32.
        valid_fft = fft_vals[1:32] 
        return valid_fft.std(), np.sum(valid_fft)

    # Apply to X, Y, Z
    x_feats = segments_df["x_segments"].apply(get_fft_features)
    y_feats = segments_df["y_segments"].apply(get_fft_features)
    z_feats = segments_df["z_segments"].apply(get_fft_features)

    # 3. FFT Std Dev
    feature_df['x_std_fft'] = x_feats.apply(lambda x: x[0])
    feature_df['y_std_fft'] = y_feats.apply(lambda x: x[0])
    feature_df['z_std_fft'] = z_feats.apply(lambda x: x[0])

    # 4. FFT SMA (Scaled by 32 as per C code)
    scaling_factor = 32.0 
    feature_df['sma_fft'] = (x_feats.apply(lambda x: x[1]) + 
                             y_feats.apply(lambda x: x[1]) + 
                             z_feats.apply(lambda x: x[1])) / scaling_factor

    labels = np.asarray(labels)
    return feature_df, labels