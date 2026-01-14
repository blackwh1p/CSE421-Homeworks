import tensorflow as tf
from tensorflow.keras.applications import MobileNet
from tensorflow.keras import layers, models

def BuildMobileNetV1(input_shape, num_classes, alpha=0.25, dropout=0.2):
    """
    Builds a MobileNetV1 architecture.
    """
    # Base Model: MobileNet V1
    # We use standard Keras implementation to ensure layer naming matches the ST model
    base_model = MobileNet(
        input_shape=input_shape,
        alpha=alpha,
        depth_multiplier=1,
        dropout=dropout,
        include_top=False, 
        weights=None 
    )
    
    # Classification Head
    x = base_model.output
    x = layers.GlobalAveragePooling2D()(x)
    if dropout > 0:
        x = layers.Dropout(dropout)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    # Final Model
    model = models.Model(inputs=base_model.input, outputs=outputs, name=f"mobilenet_v1_{alpha}")
    return model