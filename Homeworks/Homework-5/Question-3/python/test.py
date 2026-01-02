import serial
import struct
import numpy as np
import os
import time

# =======================
# CONFIGURATION
# =======================
SERIAL_PORT = 'COM6'   # <--- CHECK THIS IN DEVICE MANAGER
BAUD_RATE = 115200
DATASET_DIR = "data/MNIST-dataset"
IMG_FILE = "t10k-images.idx3-ubyte"
LBL_FILE = "t10k-labels.idx1-ubyte"
TYPE_F32 = 7
TYPE_U8  = 1

# =======================
# HELPERS
# =======================
def send_image_packet(ser, img_bytes):
    # Header: STR + Length(784) + Type(U8) + Data
    packet = b'STR'
    packet += struct.pack('<H', len(img_bytes)) 
    packet += struct.pack('<B', TYPE_U8)
    packet += img_bytes
    ser.write(packet)

def receive_result_packet(ser):
    # Wait for STW header
    while True:
        if ser.read(1) == b'S':
            if ser.read(1) == b'T':
                if ser.read(1) == b'W':
                    break
    length = struct.unpack('<H', ser.read(2))[0]
    dtype = struct.unpack('<B', ser.read(1))[0]
    raw_data = ser.read(length * 4) 
    return struct.unpack(f'<{length}f', raw_data)

def load_mnist_idx(image_path, label_path):
    if not os.path.exists(image_path): return None, None
    with open(image_path, 'rb') as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        images = np.frombuffer(f.read(), dtype=np.uint8).reshape(num, rows, cols)
    with open(label_path, 'rb') as f:
        _, num = struct.unpack(">II", f.read(8))
        labels = np.frombuffer(f.read(), dtype=np.uint8)
    return images, labels

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

# =======================
# MAIN
# =======================
if __name__ == "__main__":
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
        print(f"Connected to {SERIAL_PORT}")
    except Exception as e:
        print(f"Serial Error: {e}"); exit()

    img_path = os.path.join(DATASET_DIR, IMG_FILE)
    lbl_path = os.path.join(DATASET_DIR, LBL_FILE)
    images, labels = load_mnist_idx(img_path, lbl_path)
    
    if images is None: print("Dataset not found"); exit()

    print("\n>>> PLEASE RESET THE BOARD NOW <<<")
    print("Waiting 4 seconds for board initialization...")
    time.sleep(4)
    ser.reset_input_buffer()
    
    correct = 0
    total = 0
    NUM_TESTS = 50 

    print("-" * 65)
    print(f"{'Index':<6} | {'True':<5} | {'Pred':<5} | {'Conf':<6} | {'Result'}")
    print("-" * 65)

    for i in range(NUM_TESTS):
        img = images[i]
        true_label = labels[i]
        
        # Flatten image to 1D array of BYTES
        flat_img = img.flatten().tobytes()
        
        # 1. Send Image
        send_image_packet(ser, flat_img)
        
        # 2. Receive Result
        try:
            logits = receive_result_packet(ser)
            
            probs = softmax(np.array(logits))
            pred_label = np.argmax(probs)
            conf = probs[pred_label]
            
            status = "✅" if pred_label == true_label else "❌"
            print(f"{i:<6} | {true_label:<5} | {pred_label:<5} | {conf:.2f}   | {status}")
            
            if status == "✅": correct += 1
            total += 1
            
        except Exception as e:
            print(f"{i:<6} | Error: {e}")
            
        time.sleep(0.05)

    if total > 0:
        print("-" * 65)
        print(f"Final Accuracy: {(correct/total)*100:.1f}%")
        
    ser.close()