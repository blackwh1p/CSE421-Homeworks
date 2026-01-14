import tensorflow as tf
import numpy as np
import os
from tflite2cc import convert_tflite2cc

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(CURRENT_DIR, "model/squeezenet_tiny.h5")
OUTPUT_CC_PATH = os.path.join(CURRENT_DIR, "../hdr_cnn")

def representative_dataset():
    (_, _), (x_test, _) = tf.keras.datasets.mnist.load_data()
    for i in range(100):
        img = x_test[i:i+1]
        img = tf.image.resize(tf.expand_dims(img, -1), [32, 32])
        img = tf.image.grayscale_to_rgb(img)
        yield [tf.cast(img / 255.0, tf.float32)]

model = tf.keras.models.load_model(MODEL_PATH)
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model = converter.convert()

convert_tflite2cc(tflite_model, OUTPUT_CC_PATH)
print(f"Model Size: {len(tflite_model)/1024:.2f} KB")