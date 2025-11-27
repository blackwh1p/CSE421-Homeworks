import serial
import time
import numpy as np
from data_utils import read_data
import os.path as osp
import struct
import sys

# --- CONFIG ---
SERIAL_PORT = 'COM6'  # CHECK THIS
BAUD_RATE = 115200   # Matches C++
TIME_PERIODS = 64     # Matches C++
STEP_DISTANCE = 32

print("Loading Data...")
DATA_PATH = osp.join("WISDM_ar_v1.1", "WISDM_ar_v1.1_raw.txt")
data_df = read_data(DATA_PATH)
df_test = data_df[data_df["user"] > 28]

segments = []
labels = []

# Prepare Data (X, Y, Z arrays)
for i in range(0, len(df_test) - TIME_PERIODS, STEP_DISTANCE):
    try:
        act_slice = df_test["activity"].values[i : i + TIME_PERIODS]
        if np.all(act_slice == act_slice[0]):
            xs = df_test["x-accel"].values[i : i + TIME_PERIODS]
            ys = df_test["y-accel"].values[i : i + TIME_PERIODS]
            zs = df_test["z-accel"].values[i : i + TIME_PERIODS]
            # Python sends X array, then Y array, then Z array
            segments.append(np.concatenate([xs, ys, zs]))
            labels.append(act_slice[0])
    except:
        continue

# IMPORTANT: Sklearn classes are usually alphabetical.
# 0:Downstairs, 1:Jogging, 2:Sitting, 3:Standing, 4:Upstairs, 5:Walking
unique_labels = sorted(list(set(labels)))
label_to_int = {label: i for i, label in enumerate(unique_labels)}

print(f"Class Mapping: {label_to_int}")
print(f"Testing {len(segments)} segments...")

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
except:
    print("Error: Check COM port and close Mbed Studio monitor.")
    sys.exit()

correct = 0
total = 0
ser.reset_input_buffer()

# Initial Sync
print("Waiting for MCU sync...")
while ser.read(1) != b'R':
    pass
print("Synced!")

start_time = time.time()

for i, segment in enumerate(segments):
    # 1. Handshake (Optional for speed, but safer)
    if i > 0:
         while ser.read(1) != b'R':
             pass

    # 2. Send Data
    ser.write(segment.astype(np.float32).tobytes())

    # 3. Read Prediction
    pred = ser.read(1)
    if not pred:
        print("Timeout.")
        break
        
    pred_idx = int.from_bytes(pred, 'little')
    
    # 4. Compare
    if pred_idx == label_to_int[labels[i]]:
        correct += 1
    total += 1

print(f"Final Accuracy: {correct/total*100:.2f}%")
ser.close()