import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping

# 1. LOAD AND CLEAN DATA
df = pd.read_csv('AirQuality.csv', sep=';', decimal=',')
df = df.iloc[:, 0:15].dropna(how='all')
df.replace(-200, np.nan, inplace=True)
df.dropna(inplace=True)

# 2. FEATURE SELECTION
features = ['PT08.S1(CO)', 'PT08.S2(NMHC)', 'PT08.S3(NOx)', 
            'PT08.S4(NO2)', 'PT08.S5(O3)', 'T', 'RH', 'AH']
target = 'C6H6(GT)'
X = df[features]
y = df[target]

# 3. PREPROCESSING
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 4. DEFINE MODEL & PRINT SUMMARY
model = Sequential([
    Dense(64, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dense(32, activation='relu'),
    Dense(16, activation='relu'),
    Dense(1)
])
model.compile(optimizer='adam', loss='mse', metrics=['mae'])
model.summary() 

# 5. CONFIGURE EARLY STOPPING
early_stop = EarlyStopping(
    monitor='val_loss', 
    patience=15, 
    restore_best_weights=True,
    verbose=1
)

# 6. TRAINING
print("\nTraining for up to 500 epochs...")
history = model.fit(
    X_train_scaled, y_train, 
    epochs=500, 
    validation_split=0.2, 
    batch_size=32, 
    callbacks=[early_stop],
    verbose=1
)

# 7. EVALUATION
y_pred = model.predict(X_test_scaled)
print(f"\nFinal R-squared Score: {r2_score(y_test, y_pred):.4f}")

# 8. VISUALIZATIONS
plt.figure(figsize=(10, 5))

# Graph 1: Actual vs. Predicted
plt.subplot(1, 2, 1)
plt.scatter(y_test, y_pred, alpha=0.3, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
plt.title('Actual vs. Predicted Benzene')
plt.xlabel('Ground Truth')
plt.ylabel('Model Prediction')

# Graph 2: Loss Curve
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Train Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Training History')
plt.xlabel('Epochs')
plt.ylabel('MSE')
plt.legend()

plt.tight_layout()
plt.show()

# 9. SAVE THE MODEL
# This saves the entire model (architecture, weights, and optimizer state)
model.save('AirQuality_FCNN_Model.keras')
print("\nModel saved successfully as 'AirQuality_FCNN_Model.keras'")