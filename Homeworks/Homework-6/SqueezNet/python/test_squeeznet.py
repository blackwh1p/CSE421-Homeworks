import serial, time, numpy as np, tensorflow as tf
from tensorflow.keras.datasets import mnist

SERIAL_PORT = 'COM6' # Update to your port
BAUD_RATE = 115200

def run_test():
    (_, _), (x_test, y_test) = mnist.load_data()
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=5)
    time.sleep(2)
    print("Starting Test...")

    for i in range(20):
        img = tf.image.resize(np.expand_dims(x_test[i], -1), [32, 32])
        img = tf.image.grayscale_to_rgb(img)
        # Convert to uint8 bytes for transmission
        img_bytes = tf.cast(img, tf.uint8).numpy().flatten().tobytes()
        
        ser.reset_input_buffer()
        ser.write(img_bytes)
        
        res = ser.read(1)
        if res:
            pred = int.from_bytes(res, 'little', signed=True)
            print(f"Index {i} | True: {y_test[i]} | Pred: {pred} | {'OK' if pred==y_test[i] else 'FAIL'}")
    ser.close()

if __name__ == "__main__": run_test()