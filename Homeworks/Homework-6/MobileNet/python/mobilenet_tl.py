import os
import tensorflow as tf
from tensorflow.keras.utils import get_file, to_categorical
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import DepthwiseConv2D
from mobilenet import BuildMobileNetV1

# ---------------------------------------------------------
# KERAS VERSION PATCH
# ---------------------------------------------------------
class FixedDepthwiseConv2D(DepthwiseConv2D):
    """
    Patch to ignore 'groups=1' argument from old Keras models.
    """
    def __init__(self, **kwargs):
        if 'groups' in kwargs:
            kwargs.pop('groups')
        super().__init__(**kwargs)

# ---------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_SAVE_PATH = os.path.join(CURRENT_DIR, "model/hdr_cnn.h5")

NUM_CLASSES = 10
DATA_SHAPE = (32, 32, 3) 
ALPHA = 0.1             

ST_MODEL_URL = "https://github.com/STMicroelectronics/stm32ai-modelzoo/raw/main/image_classification/mobilenetv1/ST_pretrainedmodel_public_dataset/flowers/mobilenet_v1_0.25_96_tfs/mobilenet_v1_0.25_96_tfs.h5"

# ---------------------------------------------------------
# 1. PREPARE DATA
# ---------------------------------------------------------
print("[INFO] Loading MNIST...")
(x_train, y_train), (x_test, y_test) = mnist.load_data()

def preprocess(images):
    images = tf.expand_dims(images, axis=-1)
    images = tf.repeat(images, 3, axis=-1)
    images = tf.image.resize(images, DATA_SHAPE[:2])
    images = images / 255.0
    return images

# Using subset for speed
x_train = preprocess(x_train[:10000])
y_train = to_categorical(y_train[:10000], NUM_CLASSES)
x_test = preprocess(x_test[:1000])
y_test = to_categorical(y_test[:1000], NUM_CLASSES)

# ---------------------------------------------------------
# 2. DOWNLOAD & LOAD ST MODEL (WITH COMPILE=FALSE)
# ---------------------------------------------------------
print("[INFO] Downloading ST Model Zoo weights...")
st_model_path = get_file("mobilenet_v1_0.25_96.h5", ST_MODEL_URL, cache_subdir="models")

print("[INFO] Loading Source Model (96x96)...")

# --- FIX IS HERE: compile=False ---
# We disable compilation to avoid 'reduction=auto' error.
# We only need the weights, not the optimizer/loss state.
source_model = load_model(
    st_model_path, 
    custom_objects={'DepthwiseConv2D': FixedDepthwiseConv2D},
    compile=False 
)
print("[INFO] Source model loaded successfully!")

# ---------------------------------------------------------
# 3. TRANSFER WEIGHTS
# ---------------------------------------------------------
print(f"[INFO] Building Target Model (32x32)...")
target_model = BuildMobileNetV1(DATA_SHAPE, NUM_CLASSES, alpha=ALPHA)

print("[INFO] Transplanting weights...")
transferred_count = 0
for target_layer in target_model.layers:
    try:
        # Find layer by name
        source_layer = source_model.get_layer(target_layer.name)
        
        # Check weights compatibility
        if source_layer.weights and target_layer.weights:
            # Check shape of the main kernel (index 0)
            if source_layer.get_weights()[0].shape == target_layer.get_weights()[0].shape:
                target_layer.set_weights(source_layer.get_weights())
                target_layer.trainable = False 
                transferred_count += 1
    except:
        continue

print(f"[INFO] Transferred weights for {transferred_count} layers.")

# ---------------------------------------------------------
# 4. FINE TUNE & SAVE
# ---------------------------------------------------------
target_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("[INFO] Training...")
target_model.fit(x_train, y_train, epochs=5, batch_size=32, validation_data=(x_test, y_test))

target_model.save(MODEL_SAVE_PATH)
print(f"[SUCCESS] Model saved to: {MODEL_SAVE_PATH}")