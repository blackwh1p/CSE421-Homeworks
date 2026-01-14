import serial
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
SERIAL_PORT = 'COM6'   
BAUD_RATE = 115200
TIMEOUT_SEC = 10       

# Image Settings (Must match main.cpp)
IMG_WIDTH = 32
IMG_HEIGHT = 32
CHANNELS = 3           
IMG_SIZE = IMG_WIDTH * IMG_HEIGHT * CHANNELS

def preprocess_mnist_for_resnet(images):
    """
    Mimics the 'prepare_tensor' function from your resnet_tl.py
    1. Resizes 28x28 -> 32x32
    2. Converts Grayscale -> RGB
    3. Returns uint8 [0-255] for UART transmission
    """
    # Expand dims: (28, 28) -> (28, 28, 1)
    images = tf.expand_dims(images, axis=-1)
    # Resize: (32, 32, 1)
    images = tf.image.resize(images, [IMG_WIDTH, IMG_HEIGHT])
    # Gray to RGB: (32, 32, 3)
    images = tf.image.grayscale_to_rgb(images)
    
    # Cast to uint8 (0-255) for UART
    return tf.cast(images, tf.uint8).numpy()

def send_image_in_chunks(ser, flat_img_bytes):
    """
    Sends data in chunks to avoid STM32 buffer overflow.
    """
    CHUNK_SIZE = 32  
    total_len = len(flat_img_bytes)
    
    for i in range(0, total_len, CHUNK_SIZE):
        chunk = flat_img_bytes[i : i + CHUNK_SIZE]
        ser.write(chunk)
        time.sleep(0.005) # 5ms delay

def test_uart():
    print("[INFO] Loading MNIST Dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Test first 50 images
    num_images = 50
    raw_images = x_test[:num_images]
    y_test = y_test[:num_images]

    print("[INFO] Preprocessing images (28x28x1 -> 32x32x3)...")
    processed_images = preprocess_mnist_for_resnet(raw_images)

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT_SEC)
        print(f"[INFO] Connected to {SERIAL_PORT}")
        time.sleep(2) 
    except Exception as e:
        print(f"[ERROR] Serial error: {e}")
        return

    correct_count = 0
    print("-" * 50)
    print(f"{'Index':<10} | {'True':<10} | {'Pred':<10} | {'Status'}")
    print("-" * 50)

    for i in range(num_images):
        # Flatten the 32x32x3 image to bytes
        flat_img = processed_images[i].flatten().tobytes()

        # Clear buffer
        ser.flushInput() 
        
        # Send Data
        send_image_in_chunks(ser, flat_img)

        # Wait for Response
        response = ser.read(1)
        
        true_label = int(y_test[i])
        status = "FAIL"
        pred_label = "TIMEOUT"

        if len(response) == 1:
            pred_label = int.from_bytes(response, byteorder='little', signed=True)
            if pred_label == true_label:
                status = "PASS"
                correct_count += 1
        
        print(f"{i:<10} | {true_label:<10} | {pred_label:<10} | {status}")

    accuracy = (correct_count / num_images) * 100
    print("-" * 50)
    print(f"[RESULT] Accuracy: {accuracy:.2f}%")
    
    ser.close()

if __name__ == "__main__":
    test_uart()