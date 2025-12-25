import os
import pandas as pd
import numpy as np
import keras
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

# --- UPDATED PATH CONFIGURATION ---
# Points directly to the 'data' and 'Models' folders in your current directory
current_dir = os.path.dirname(os.path.abspath(__file__))
TEMPERATURE_DATA_PATH = os.path.join(current_dir, "data", "temperature_dataset.csv")
KERAS_MODEL_DIR = os.path.join(current_dir, "Models")

if not os.path.exists(KERAS_MODEL_DIR):
    os.makedirs(KERAS_MODEL_DIR)

# 1. Load the Filtered CSV Data
df = pd.read_csv(TEMPERATURE_DATA_PATH)

# Use the Room_Temp column. [::4] downsamples from 15-min to 1-hour intervals
y = df['Room_Temp'][::4]
prev_values_count = 5

# 2. Windowing Logic (Time-Series Features)
X = pd.DataFrame()
for i in range(prev_values_count, 0, -1):
    X['t-' + str(i)] = y.shift(i)

# Remove the empty rows created by shifting
X = X[prev_values_count:]
y = y[prev_values_count:]

# 3. Data Split and Normalization
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

train_mean = X_train.mean()
train_std = X_train.std()

X_train = (X_train - train_mean) / train_std
X_test = (X_test - train_mean) / train_std

# 4. Model Definition (Application 11.9)
model = keras.models.Sequential([
    keras.layers.Dense(100, input_shape=[5], activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(1) # Final estimation output
])

# 5. Compilation and Training
model.compile(optimizer=keras.optimizers.SGD(learning_rate=5e-4),
              loss=keras.losses.MeanAbsoluteError())

print("Starting training...")
model.fit(X_train,
          y_train, 
          batch_size=128, 
          epochs=3000, 
          verbose=1)

# 6. Prediction and Visualization
y_train_predicted = model.predict(X_train)
y_test_predict = model.predict(X_test)

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(y_test.to_numpy(), label="Actual values")
ax.plot(y_test_predict, label="Predicted values")
plt.title("Application 11.9: Future Temperature Estimation")
plt.legend()
plt.show()

# 7. Metrics
# Calculating the root of MAE as per your original request
mae_train = np.sqrt(mean_absolute_error(y_train, y_train_predicted))
mae_test = np.sqrt(mean_absolute_error(y_test, y_test_predict))

print(f"\nTraining set MAE (Sqrt): {mae_train:.4f}")
print(f"Test set MAE (Sqrt): {mae_test:.4f}")

# 8. Save Model
model_save_path = os.path.join(KERAS_MODEL_DIR, "temperature_pred_mlp.h5")
model.save(model_save_path)
print(f"Model saved to: {model_save_path}")