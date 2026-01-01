import serial
import struct
import numpy as np
import pandas as pd
import time

# =======================
# Configuration
# =======================
SERIAL_PORT = 'COM6'
BAUD_RATE = 115200

DATASET_PATH = 'data/WISDM_ar_v1.1_raw.txt'

WINDOW_SIZE = 80
STEP_SIZE = 40
NUM_CLASSES = 6

ACTIVITY_MAP = {
    "Downstairs": 0,
    "Jogging": 1,
    "Sitting": 2,
    "Standing": 3,
    "Upstairs": 4,
    "Walking": 5
}

# =======================
# Dataset utilities
# =======================
def load_data(file_path):
    print("Loading WISDM dataset...")

    column_names = [
        'user', 'activity', 'timestamp',
        'x-axis', 'y-axis', 'z-axis'
    ]

    df = pd.read_csv(
        file_path,
        header=None,
        names=column_names,
        sep=',',
        engine='python',
        on_bad_lines='skip'
    )

    # Clean z-axis
    df['z-axis'] = df['z-axis'].astype(str).str.replace(';', '', regex=False)

    # Convert numeric columns
    df[['user', 'timestamp', 'x-axis', 'y-axis', 'z-axis']] = (
        df[['user', 'timestamp', 'x-axis', 'y-axis', 'z-axis']]
        .apply(pd.to_numeric, errors='coerce')
    )

    # Drop rows with invalid numeric data
    df.dropna(inplace=True)

    # Map activity strings to labels
    df = df[df['activity'].isin(ACTIVITY_MAP.keys())]
    df['activity'] = df['activity'].map(ACTIVITY_MAP)

    df['user'] = df['user'].astype(int)
    df['activity'] = df['activity'].astype(int)

    print(f"Loaded {len(df)} valid samples")
    print("Activities:", sorted(df['activity'].unique()))

    return df


def get_segments(df, window_size, step_size):
    segments = []
    labels = []

    for i in range(0, len(df) - window_size, step_size):
        xs = df['x-axis'].values[i:i + window_size]
        ys = df['y-axis'].values[i:i + window_size]
        zs = df['z-axis'].values[i:i + window_size]

        segment = np.array([xs, ys, zs], dtype=np.float32)
        label = df['activity'].iloc[i]

        segments.append(segment)
        labels.append(label)

    return segments, labels


# =======================
# Main
# =======================
if __name__ == "__main__":

    df = load_data(DATASET_PATH)

    # Use unseen users as test data
    test_df = df[df['user'] >= 29]

    segments, labels = get_segments(
        test_df, WINDOW_SIZE, STEP_SIZE
    )

    if len(segments) == 0:
        raise RuntimeError(
            "❌ No segments generated. "
            "Check WINDOW_SIZE / STEP_SIZE or dataset filtering."
        )

    print(f"Generated {len(segments)} segments")

    print(f"Connecting to Mbed on {SERIAL_PORT}...")
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=3)
    time.sleep(2)

    total_samples = min(100, len(segments))
    correct = 0

    print(f"Starting inference on {total_samples} samples...\n")

    for i in range(total_samples):

        input_data = segments[i].flatten()
        ser.write(input_data.tobytes())

        response_size = NUM_CLASSES * 4
        response = ser.read(response_size)

        if len(response) != response_size:
            print(f"[{i}] ❌ Timeout")
            continue

        probs = struct.unpack(f'{NUM_CLASSES}f', response)
        pred = int(np.argmax(probs))
        conf = probs[pred]

        true_label = labels[i]

        if pred == true_label:
            correct += 1

        print(
            f"[{i:03d}] "
            f"Pred={pred} (conf={conf:.2f}) | "
            f"True={true_label} | "
            f"{'✔' if pred == true_label else '✘'}"
        )

    ser.close()

    accuracy = correct / total_samples * 100.0
    print("\n======================")
    print(f"Accuracy: {accuracy:.2f}%")
    print("======================")
