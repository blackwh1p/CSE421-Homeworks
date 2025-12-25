import os
import sys

# --- 1. Suppress Warnings ---
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import warnings
warnings.filterwarnings("ignore")

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

def main():
    print("Initializing Model...")

    # 1. Define Weights and Biases
    # We want 1 neuron with 2 inputs.
    # Weights shape must be (Input_Dim, Units) -> (2, 1)
    # This ensures we get exactly 1 output per sample.
    w_init = tf.constant_initializer([[0.5], [-0.5]])
    b_init = tf.constant_initializer([0.0])

    # 2. Create Model
    # Using an explicit Input layer fixes the 'UserWarning'
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(2,)), # Input has 2 features (x and y)
        tf.keras.layers.Dense(
            units=1,                       # ONE output (important for plotting)
            kernel_initializer=w_init, 
            bias_initializer=b_init, 
            activation='sigmoid'
        )
    ])

    # 3. Generate Data Grid
    # Range -5 to 5 with step 0.1 = 100 points per axis
    # 100 * 100 = 10,000 total data points
    x_range = np.arange(-5, 5, 0.1)
    y_range = np.arange(-5, 5, 0.1)
    x_grid, y_grid = np.meshgrid(x_range, y_range)

    # Flatten data to shape (10000, 2) for the model
    x_flat = x_grid.ravel()
    y_flat = y_grid.ravel()
    input_data = np.column_stack((x_flat, y_flat))

    # 4. Run Inference
    print(f"Calculating output for {input_data.shape[0]} points...")
    Z = model.predict(input_data, verbose=0)
    
    # 5. Reshape for Plotting
    # Z comes out as shape (10000, 1). We flatten it to (10000,) 
    # so it can be reshaped to (100, 100)
    Z = Z.flatten()
    
    if Z.size != x_grid.size:
        print(f"Error: Model produced {Z.size} outputs, but grid needs {x_grid.size}.")
        print("Check if 'units' in Dense layer is set to 1.")
        return

    Z_reshaped = Z.reshape(x_grid.shape)

    # 6. Plot
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(10, 8))
    surf = ax.plot_surface(x_grid, y_grid, Z_reshaped, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    
    ax.set_xlabel('Input X')
    ax.set_ylabel('Input Y')
    ax.set_zlabel('Sigmoid Output')
    ax.set_title('Single Neuron Activation')
    fig.colorbar(surf, shrink=0.5, aspect=5)
    
    print("Displaying plot...")
    plt.show()

if __name__ == "__main__":
    main()