import tensorflow as tf
import numpy as np
import cv2
import os
import random

# --- SETTINGS ---
MODEL_PATH = "models/plant_diseases_mobilenet.h5"
TEST_DIR = "dataset/test" # Kaggle dataset has a 'test' folder
IMG_SIZE = 96

# --- LOAD MODEL ---
if not os.path.exists(MODEL_PATH):
    print("Model not found. Run training.py first.")
    exit()

print("[INFO] Loading Model...")
model = tf.keras.models.load_model(MODEL_PATH)

# --- GET CLASS NAMES ---
# We read class names from the test folder to ensure they match training
class_names = sorted([d for d in os.listdir(TEST_DIR) if os.path.isdir(os.path.join(TEST_DIR, d))])
print(f"[INFO] Classes: {class_names}")

# --- TEST ON RANDOM IMAGES ---
print("\n--- STARTING PREDICTIONS ---")
print("Press 'q' to quit, space for next image.")

# Collect all test images
all_test_images = []
for label in class_names:
    folder_path = os.path.join(TEST_DIR, label)
    for file in os.listdir(folder_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')):
            all_test_images.append((os.path.join(folder_path, file), label))

# Shuffle to test random fruits
random.shuffle(all_test_images)

for img_path, true_label in all_test_images:
    # Preprocess
    img = cv2.imread(img_path)
    if img is None: continue
    
    # Resize to 64x64
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    
    # Expand dims (1, 64, 64, 3) - Note: Rescaling is inside the model layer
    input_data = np.expand_dims(img_resized, axis=0)

    # Predict
    predictions = model.predict(input_data, verbose=0)
    score = tf.nn.softmax(predictions[0]) # Get probabilities
    
    pred_index = np.argmax(predictions[0])
    pred_label = class_names[pred_index]
    confidence = 100 * np.max(score)

    # Visualization
    # Resize for better viewing on screen
    display_img = cv2.resize(img, (400, 400))
    
    # Text Color: Green if correct, Red if wrong
    color = (0, 255, 0) if pred_label == true_label else (0, 0, 255)
    
    text = f"True: {true_label}"
    text2 = f"Pred: {pred_label} ({confidence:.1f}%)"
    
    cv2.putText(display_img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(display_img, text2, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    print(f"File: {os.path.basename(img_path)} | True: {true_label} | Pred: {pred_label}")

    cv2.imshow("Fruit/Veg Recognition", display_img)
    
    key = cv2.waitKey(0) & 0xFF
    if key == ord('q'):
        break

cv2.destroyAllWindows()