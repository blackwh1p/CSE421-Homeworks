import os
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
from squeezenet import SqueezeNet

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILE = os.path.join(CURRENT_DIR, "model/squeezenet_tiny.h5")
if not os.path.exists(os.path.join(CURRENT_DIR, "model")): os.makedirs(os.path.join(CURRENT_DIR, "model"))

def prepare_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    def preprocess(imgs):
        imgs = tf.expand_dims(imgs, -1)
        imgs = tf.image.resize(imgs, [32, 32])
        imgs = tf.image.grayscale_to_rgb(imgs)
        return imgs / 255.0
    return preprocess(x_train), to_categorical(y_train, 10), preprocess(x_test), to_categorical(y_test, 10)

if __name__ == "__main__":
    x_train, y_train, x_test, y_test = prepare_data()
    model = SqueezeNet(input_shape=(32, 32, 3), classes=10)
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    model.fit(x_train, y_train, epochs=5, batch_size=64, validation_data=(x_test, y_test))
    model.save(MODEL_FILE)