import serial
import time
import numpy as np
import struct
import sys
import os
from mnist import load_images, load_labels

# --- CONFIG ---
SERIAL_PORT = 'COM6' 
BAUD_RATE = 115200
# NOTE: Use the exact same IMG_SIZE as defined in your C code
IMG_SIZE = 784 

print("Loading Test Data...")
DATA_DIR = "MNIST-dataset"
test_images = load_images(os.path.join(DATA_DIR, "t10k-images.idx3-ubyte"))
test_labels = load_labels(os.path.join(DATA_DIR, "t10k-labels.idx1-ubyte"))

# Test 500 images
indices = np.random.choice(len(test_images), 500, replace=False)

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=2)
except:
    print("Error opening serial port")
    sys.exit()

ser.reset_input_buffer()
print("Waiting for MCU...")
while ser.read(1) != b'R':
    pass
print("Synced!")

correct = 0
total = 0

for i, idx in enumerate(indices):
    img = test_images[idx]
    label = test_labels[idx]

    # 1. Handshake
    if i > 0:
        while ser.read(1) != b'R':
            pass

    # 2. Send Image (THE FIX)
    # The MCU C-code reads data in Column-Major order (c * height + r).
    # Python stores data in Row-Major order.
    # We must Transpose (.T) the image so the byte stream matches the MCU's reading logic.
    # This ensures the MCU calculates features on a "Normal 7", not a "Flipped 7".
    
    img_for_mcu = img.T  # <--- THIS IS THE KEY FIX
    ser.write(img_for_mcu.tobytes())

    # 3. Receive Prediction
    resp = ser.read(1)
    if not resp:
        break
    
    pred = int.from_bytes(resp, 'little')
    
    if pred == label:
        correct += 1
    total += 1
    
    if i % 50 == 0:
        print(f"Processed {i} | Acc: {correct/total*100:.2f}%")

print("-" * 30)
print(f"Final Accuracy: {correct/total*100:.2f}%")
print("-" * 30)
ser.close()