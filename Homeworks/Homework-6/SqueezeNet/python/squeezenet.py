import tensorflow as tf
from keras.utils import get_file

WEIGHTS_PATH_NO_TOP = "https://github.com/rcmalli/keras-squeezenet/releases/download/v1.0/squeezenet_weights_tf_dim_ordering_tf_kernels_notop.h5"

def fire(x, squeeze, expand, name):
    y = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, activation="relu", padding="same", name=f"{name}_s")(x)
    y1 = tf.keras.layers.Conv2D(filters=expand, kernel_size=1, activation="relu", padding="same", name=f"{name}_e1")(y)
    y3 = tf.keras.layers.Conv2D(filters=expand, kernel_size=3, activation="relu", padding="same", name=f"{name}_e3")(y)
    return tf.keras.layers.concatenate([y1, y3], name=f"{name}_c")

def SqueezeNet(input_shape=(32, 32, 3), weights="imagenet", classes=10, dropout=0.2):
    model_input = tf.keras.layers.Input(shape=input_shape)
    
    # Giriş Katmanı
    x = tf.keras.layers.Conv2D(64, (3, 3), strides=2, padding="same", activation="relu", name="conv1")(model_input)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)
    
    # STM32 için optimize edilmiş derinlik (Fire 4'e kadar)
    x = fire(x, 16, 64, name="fire1")
    x = fire(x, 16, 64, name="fire2")
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)
    x = fire(x, 32, 128, name="fire3")
    x = fire(x, 32, 128, name="fire4")

    if weights == "imagenet":
        weights_path = get_file("sq_weights_notop.h5", WEIGHTS_PATH_NO_TOP, cache_subdir="models")
        # Sadece mevcut katmanların ağırlıklarını yükle
        tmp_model = tf.keras.Model(inputs=model_input, outputs=x)
        tmp_model.load_weights(weights_path, by_name=True, skip_mismatch=True)

    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv2D(classes, (1, 1), padding="same", name="final_conv")(x)
    x = tf.keras.layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    model_output = tf.keras.layers.Softmax()(x)
    
    return tf.keras.Model(inputs=model_input, outputs=model_output)