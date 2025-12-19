import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
# IMPORTANT: We import from sklearn2c instead of sklearn.linear_model
from sklearn2c import LinearRegressor 
from sklearn.metrics import mean_absolute_error
from matplotlib import pyplot as plt

# --- 1. SETUP PATHS ---
script_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(script_dir, "dataset/temperature_dataset.csv")
model_save_path = os.path.join(script_dir, "../temperature_pred_linreg.joblib")

# --- 2. LOAD DATA ---
try:
    df = pd.read_csv(csv_path)
    y = df["Room_Temp"][::4] # Resample to 1-hour intervals
    prev_values_count = 5

    X = pd.DataFrame()
    for i in range(prev_values_count, 0, -1):
        X["t-" + str(i)] = y.shift(i)

    X = X[prev_values_count:]
    y = y[prev_values_count:]
except Exception as e:
    print(f"Error loading CSV: {e}")
    exit()

# --- 3. TRAIN MODEL ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# Use the sklearn2c version of the model
model = LinearRegressor()

# sklearn2c uses the .train() method (as shown in book Listing 7.1)
# This fits the model AND saves it to the path automatically
print("Training model...")
model.train(X_train.to_numpy(), y_train.to_numpy(), save_path=model_save_path)

# --- 4. EVALUATION & PLOT ---
y_test_predict = model.predict(X_test.to_numpy())

plt.figure(figsize=(10, 5))
plt.plot(y_test.to_numpy()[:100], label="Actual", color="black")
plt.plot(y_test_predict[:100], label="Predicted", color="blue", linestyle="--")
plt.title("Temperature Prediction Training Success")
plt.legend()
plt.show()

mae_test = mean_absolute_error(y_test, y_test_predict)
print(f"Test set Error (RMSE): {np.sqrt(mae_test):.4f}")
print(f"Model saved correctly for sklearn2c at: {model_save_path}")