import os
import numpy as np
import pandas as pd
import serial
import struct
import time
from sklearn2c import LinearRegressor

# --- CONFIG ---
PORT = "COM6" # Double check this!
BAUD = 115200
script_dir = os.path.dirname(os.path.abspath(__file__))
model_path = os.path.join(script_dir, "../temperature_pred_linreg.joblib")
csv_path = os.path.join(script_dir, "dataset/temperature_dataset.csv")

# --- LOAD DATA ---
print("Loading model and dataset...")
linear = LinearRegressor.load(model_path)
df = pd.read_csv(csv_path)
y_data = df["Room_Temp"][::4]
test_samples = []
for i in range(5, 0, -1):
    test_samples.append(y_data.shift(i))
test_samples = pd.concat(test_samples, axis=1)[5:].to_numpy()

# --- OPEN SERIAL ---
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    ser.flushInput()
    ser.flushOutput()
    print(f"Connected to {PORT}. If it hangs, press the RESET button on the board.")
except Exception as e:
    print(f"Error: {e}")
    exit()

i = 0
pc_prediction = 0.0

try:
    while True:
        # 1. Look for the first character of a header ('S')
        char = ser.read(1)
        if char != b'S':
            continue 
        
        # 2. Read the next 2 chars to see if it is "TR" or "TW"
        remaining_header = ser.read(2)
        header = "S" + remaining_header.decode('ascii', errors='ignore')

        if header not in ["STR", "STW"]:
            continue # Not a valid packet, keep looking

        # 3. Read Type (1 byte) and Length (4 bytes)
        data_type = int.from_bytes(ser.read(1), byteorder='little')
        raw_len = ser.read(4)
        if len(raw_len) < 4: continue
        byte_length = struct.unpack('<I', raw_len)[0]

        # --- SAFETY CHECK ---
        # If byte_length is huge, the serial is out of sync. Skip it.
        if byte_length > 5000:
            ser.flushInput()
            continue

        if header == "STR":
            # MCU asks for data
            sample = test_samples[i].astype(np.float32)
            ser.write(sample.tobytes())
            
            # PC Prediction
            pred = linear.predict(sample.reshape(1, -1))
            pc_prediction = float(np.ravel(pred)[0])
            i = (i + 1) % len(test_samples)

        elif header == "STW":
            # MCU sends result
            payload = ser.read(byte_length)
            if len(payload) == 4:
                mcu_prediction = struct.unpack('f', payload)[0]
                print(f"Sample {i}: PC: {pc_prediction:.4f} | MCU: {mcu_prediction:.4f} | {'MATCH ✅' if abs(pc_prediction-mcu_prediction)<0.05 else 'ERROR ❌'}")

except KeyboardInterrupt:
    print("\nUser stopped the script.")
finally:
    ser.close()