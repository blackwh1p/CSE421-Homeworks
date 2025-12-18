import os
import numpy as np
import pandas as pd
import serial
import struct
from sklearn2c import LinearRegressor

# --- CONFIG ---
PORT = "COM6"  # Ensure this is correct
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

total_samples = len(test_samples)
print(f"Total samples to process: {total_samples}")

# --- OPEN SERIAL ---
try:
    ser = serial.Serial(PORT, BAUD, timeout=1)
    ser.flushInput()
    ser.flushOutput()
    print(f"Connected to {PORT}. Waiting for MCU...")
except Exception as e:
    print(f"Error: {e}")
    exit()

# --- PROCESSING LOOP ---
# We use range(total_samples) so it stops when the data ends
for i in range(total_samples):
    found_packet = False
    pc_prediction = 0.0

    while not found_packet:
        char = ser.read(1)
        if char == b'S':
            suffix = ser.read(2)
            header = "S" + suffix.decode('ascii', errors='ignore')
            
            if header == "STR":
                # 1. Send Data to MCU
                sample = test_samples[i].astype(np.float32)
                ser.write(sample.tobytes())
                
                # 2. Calculate PC Prediction
                pred = linear.predict(sample.reshape(1, -1))
                pc_prediction = float(np.ravel(pred)[0])
                
                # 3. Wait for MCU to send result back (STW)
                # We loop briefly to find the STW response
                for _ in range(100): 
                    res_char = ser.read(1)
                    if res_char == b'S':
                        res_suffix = ser.read(2)
                        if res_suffix == b'TW':
                            ser.read(1) # skip type
                            ser.read(4) # skip length
                            payload = ser.read(4)
                            mcu_prediction = struct.unpack('f', payload)[0]
                            
                            # 4. Print Comparison
                            status = "MATCH ✅" if abs(pc_prediction - mcu_prediction) < 0.05 else "MISMATCH ❌"
                            print(f"Sample {i+1}/{total_samples}: PC: {pc_prediction:.4f} | MCU: {mcu_prediction:.4f} | {status}")
                            
                            found_packet = True
                            break
    
print("\n" + "="*30)
print("TEST COMPLETE: All samples processed.")
print("="*30)
ser.close()