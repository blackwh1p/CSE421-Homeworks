import os
import tensorflow as tf
from squeezenet import SqueezeNet

# Hardcoded paths replacing Models.paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
KERAS_MODEL_DIR = os.path.join(CURRENT_DIR, "model")
if not os.path.exists(KERAS_MODEL_DIR): os.makedirs(KERAS_MODEL_DIR)

model_checkpoint_path = os.path.join(KERAS_MODEL_DIR, "squeezenet_tl_mnist.h5")
num_classes = 10
data_shape = (32, 32, 3)

def prepare_tensor(images, out_shape):
    images = tf.expand_dims(images, axis=-1)
    images = tf.repeat(images, 3, axis=-1)
    images = tf.image.resize(images, out_shape[:2])
    images = images / 255.0
    return images

if __name__ == "__main__":
    print("[INFO] Loading and preparing MNIST dataset...")
    (train_images, train_labels), (test_images, test_labels) = tf.keras.datasets.mnist.load_data()

    train_images = prepare_tensor(train_images, data_shape)
    test_images = prepare_tensor(test_images, data_shape)

    train_labels = tf.keras.utils.to_categorical(train_labels, num_classes)
    test_labels = tf.keras.utils.to_categorical(test_labels, num_classes)

    model = SqueezeNet(input_shape=data_shape, dropout=0.2)
    
    # Freeze initial layers for Transfer Learning
    num_layers_to_train = len(model.layers) // 3
    for layer in model.layers[:num_layers_to_train]:
        layer.trainable = False

    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            model_checkpoint_path,
            monitor="val_loss",
            save_best_only=True,
            mode="min",
            verbose=1,
        )
    ]

    model.fit(
        train_images,
        train_labels,
        batch_size=128,
        epochs=10,
        validation_split=0.1,
        callbacks=callbacks,
    )
    print(f"[SUCCESS] Model saved to {model_checkpoint_path}")