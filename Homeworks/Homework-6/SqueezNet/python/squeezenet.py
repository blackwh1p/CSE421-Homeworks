import tensorflow as tf

def fire(x, squeeze, expand, name):
    y = tf.keras.layers.Conv2D(filters=squeeze, kernel_size=1, activation="relu", padding="same", name=f"{name}_squeeze")(x)
    y1 = tf.keras.layers.Conv2D(filters=expand, kernel_size=1, activation="relu", padding="same", name=f"{name}_expand1x1")(y)
    y3 = tf.keras.layers.Conv2D(filters=expand, kernel_size=3, activation="relu", padding="same", name=f"{name}_expand3x3")(y)
    return tf.keras.layers.concatenate([y1, y3], name=f"{name}_concat")

def SqueezeNet(input_shape=(32, 32, 3), classes=10, dropout=0.2):
    img_input = tf.keras.layers.Input(shape=input_shape)
    
    # Reduced Stem
    x = tf.keras.layers.Conv2D(32, (3, 3), strides=2, padding="same", activation="relu", name="conv1")(img_input)
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)
    
    # Tiny Fire Stack (4 blocks only)
    x = fire(x, 8, 32, name="fire1")
    x = fire(x, 8, 32, name="fire2")
    x = tf.keras.layers.MaxPooling2D((3, 3), strides=2, padding="same")(x)
    x = fire(x, 16, 64, name="fire3")
    x = fire(x, 16, 64, name="fire4")

    if dropout:
        x = tf.keras.layers.Dropout(dropout)(x)

    x = tf.keras.layers.Conv2D(classes, (1, 1), name="final_conv")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    output = tf.keras.layers.Softmax()(x)
    
    return tf.keras.Model(inputs=img_input, outputs=output, name="TinySqueezeNet")