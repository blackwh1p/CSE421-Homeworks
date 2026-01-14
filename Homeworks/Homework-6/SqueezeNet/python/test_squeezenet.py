import serial
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

# ---------------------------------------------------------
# CONFIGURATION (Restored from your original test_resnet.py)
# ---------------------------------------------------------
SERIAL_PORT = 'COM6'   
BAUD_RATE = 115200
TIMEOUT_SEC = 10       

# Image Settings (Must match main.cpp)
IMG_WIDTH = 32
IMG_HEIGHT = 32
CHANNELS = 3           
IMG_SIZE = IMG_WIDTH * IMG_HEIGHT * CHANNELS # 3072 bytes

def preprocess_mnist_for_squeezenet(images):
    """
    Matches your original preprocessing logic:
    1. Resizes 28x28 -> 32x32
    2. Converts Grayscale -> RGB
    3. Returns uint8 [0-255] for UART transmission
    """
    images = tf.expand_dims(images, axis=-1)
    images = tf.image.resize(images, [IMG_WIDTH, IMG_HEIGHT])
    images = tf.image.grayscale_to_rgb(images)
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
        time.sleep(0.005) # 5ms delay to ensure UART stability

def test_uart():
    print("[INFO] Loading MNIST Dataset...")
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Test first 50 images (Matching your original request)
    num_images = 50
    raw_images = x_test[:num_images]
    y_test_subset = y_test[:num_images]

    print("[INFO] Preprocessing images (28x28x1 -> 32x32x3)...")
    processed_images = preprocess_mnist_for_squeezenet(raw_images)

    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=TIMEOUT_SEC)
        print(f"[INFO] Connected to {SERIAL_PORT}")
        time.sleep(2) # Wait for STM32 to finish InitModel()
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
        ser.reset_input_buffer() 
        
        # Send Data in chunks
        send_image_in_chunks(ser, flat_img)

        # Wait for Response (1 byte)
        response = ser.read(1)
        
        true_label = int(y_test_subset[i])
        status = "FAIL"
        pred_label = "TIMEOUT"

        if len(response) == 1:
            # Interpret byte as signed integer to match int8_t from Mbed
            pred_label = int.from_bytes(response, byteorder='little', signed=True)
            if pred_label == true_label:
                status = "PASS"
                correct_count += 1
        
        print(f"{i:<10} | {true_label:<10} | {pred_label:<10} | {status}")

    # Accuracy Calculation (Restored from your original code)
    accuracy = (correct_count / num_images) * 100
    print("-" * 50