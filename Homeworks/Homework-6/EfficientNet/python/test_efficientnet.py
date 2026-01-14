import serial
import time
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

SERIAL_PORT = 'COM6'   
BAUD_RATE = 115200

def test_uart():
    (_, _), (x_test, y_test) = mnist.load_data()
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=15)
    time.sleep(2)

    for i in range(20):
        img = x_test[i]
        # Preprocess for transmission (Raw uint8)
        img_p = tf.image.resize(np.expand_dims(img, -1), [32, 32])
        img_p = tf.image.grayscale_to_rgb(img_p)
        img_bytes = tf.cast(img_p, tf.uint8).numpy().flatten().tobytes()

        ser.reset_input_buffer()
        ser.write(img_bytes)
        
        response = ser.read(1)
        if response:
            pred = int.from_bytes(response, 'little', signed=True)
            print(f"Index {i} | True: {y_test[i]} | Pred: {pred} | {'PASS' if pred==y_test[i] else 'FAIL'}")
    ser.close()

if __name__ == "__main__":
    test_uart()