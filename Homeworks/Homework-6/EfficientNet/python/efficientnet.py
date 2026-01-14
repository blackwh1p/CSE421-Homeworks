import tensorflow as tf
import keras
from keras import layers, ops
from typing import List, Tuple
import math
from keras.models import Model

# Default EfficientNet repeats
repeats = [1, 2, 2, 3, 3, 4, 1]

def round_expansion(expansion_factor: int, repeats: List[int]) -> List[int]:
    exp_ratio = []
    flag = 1
    for r in repeats:
        if (r != 0) and flag:
            exp_ratio.append(1)
            flag = 0
        else:
            exp_ratio.append(expansion_factor)
    return exp_ratio

def round_filters(filters: int, width_coefficient: float, depth_divisor: int = 8, min_filters: int = None) -> int:
    if not width_coefficient:
        return filters
    filters *= width_coefficient
    min_filters = min_filters or depth_divisor
    new_filters = max(int(filters + depth_divisor / 2) // depth_divisor * depth_divisor, min_filters)
    if new_filters < 0.9 * filters:
        new_filters += depth_divisor
    return int(new_filters)

def round_repeats(repeats: List[int]) -> List[int]:
    num_repeat = sum(repeats)
    num_repeat_scaled = int(math.ceil(num_repeat))
    repeats_scaled = []
    for r in repeats[::-1]:
        rs = max(1, round((r / num_repeat * num_repeat_scaled)))
        repeats_scaled.append(rs)
        num_repeat -= r
        num_repeat_scaled -= rs
    repeats_scaled = repeats_scaled[::-1]
    return repeats_scaled    

def mb_conv_block(inputs, in_channels, out_channels, num_repeat, stride, expansion_factor, se_ratio, k, drop_rate, prev_block_num, activation):
    x = inputs
    input_filters = in_channels
	
    for i in range(num_repeat):
        input_tensor = x
        current_stride = stride if i == 0 else 1
        expanded_filters = input_filters * expansion_factor

        # Expansion Phase
        if expansion_factor != 1:
            x = layers.Conv2D(filters=expanded_filters, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(x)
            x = layers.BatchNormalization()(x)
            x = layers.Activation(activation)(x)

        # Depthwise Phase
        x = layers.DepthwiseConv2D(kernel_size=(k, k), strides=current_stride, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation(activation)(x)

        # Squeeze and Excitation (SE)
        squeezed_filters = max(1, int(input_filters * se_ratio))
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Reshape((1, 1, expanded_filters))(se)
        se = layers.Conv2D(filters=squeezed_filters, kernel_size=(1, 1), padding='same', activation=activation)(se)
        se = layers.Conv2D(filters=expanded_filters, kernel_size=(1, 1), padding='same')(se)
        
        # FIXED: Use ops.clip instead of manual Minimum/Maximum to handle symbolic shapes
        se = ops.clip(se, -4.0, 4.0)
        se = layers.Activation('sigmoid')(se)
        x = layers.Multiply()([x, se])
        
        # Output Projection
        x = layers.Conv2D(filters=out_channels, kernel_size=(1, 1), strides=1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization()(x)

        # Skip Connection
        if current_stride == 1 and input_filters == out_channels:
            if drop_rate > 0:
                x = layers.Dropout(drop_rate)(x)
            x = layers.Add()([x, input_tensor])
            
        input_filters = out_channels
    return x

def EfficientNet(input_shape=(32,32,3), classes=10, dropout_rate=0.2, se_ratio=0.25, drop_connect_rate=0.2):
    img_input = layers.Input(shape=input_shape)
    coeff = 0.45 

    # Stem
    x = layers.Conv2D(filters=round_filters(32, coeff), kernel_size=(3, 3), strides=(2, 2), padding='same', use_bias=False)(img_input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)

    # Scaled Blocks
    r_scaled = round_repeats(repeats)
    e_ratio = round_expansion(3, r_scaled)
    
    # Blocks 1-7
    x = mb_conv_block(x, round_filters(32, coeff), round_filters(16, coeff), r_scaled[0], 1, e_ratio[0], se_ratio, 3, drop_connect_rate, 0, "relu6")
    x = mb_conv_block(x, round_filters(16, coeff), round_filters(24, coeff), r_scaled[1], 2, e_ratio[1], se_ratio, 3, drop_connect_rate, sum(r_scaled[:1]), "relu6")
    x = mb_conv_block(x, round_filters(24, coeff), round_filters(40, coeff), r_scaled[2], 2, e_ratio[2], se_ratio, 5, drop_connect_rate, sum(r_scaled[:2]), "relu6")
    x = mb_conv_block(x, round_filters(40, coeff), round_filters(80, coeff), r_scaled[3], 2, e_ratio[3], se_ratio, 3, drop_connect_rate, sum(r_scaled[:3]), "relu6")
    x = mb_conv_block(x, round_filters(80, coeff), round_filters(112, coeff), r_scaled[4], 1, e_ratio[4], se_ratio, 5, drop_connect_rate, sum(r_scaled[:4]), "relu6")
    x = mb_conv_block(x, round_filters(112, coeff), round_filters(192, coeff), r_scaled[5], 2, e_ratio[5], se_ratio, 5, drop_connect_rate, sum(r_scaled[:5]), "relu6")
    x = mb_conv_block(x, round_filters(192, coeff), round_filters(320, coeff), r_scaled[6], 1, e_ratio[6], se_ratio, 3, drop_connect_rate, sum(r_scaled[:6]), "relu6")

    # Head
    x = layers.Conv2D(filters=round_filters(1280, coeff), kernel_size=(1, 1), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu6")(x)
	
    x = layers.GlobalAveragePooling2D()(x)
    if dropout_rate > 0:
        x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(classes, activation='softmax')(x)

    return Model(img_input, x, name="EfficientNet-STM32")