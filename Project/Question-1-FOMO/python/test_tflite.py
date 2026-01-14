import tensorflow as tf
import numpy as np
import cv2
import os
import json

# --- SETTINGS ---
MODEL_PATH = "models/fruitfly_fomo.h5"
TEST_IMAGE_DIR = "dataset/test"
LABEL_FILE = "dataset/test_labels.json"
IMG_SIZE = 96
GRID_SIZE = 12
ORIGINAL_SIZE = 400
THRESHOLD = 0.5  # Eğer Weighted Loss kullandıysak, bu değer güvenilirdir.

if not os.path.exists(MODEL_PATH):
    print("Run training.py first!")
    exit()

# Load model without compiling (to avoid loss function error)
model = tf.keras.models.load_model(MODEL_PATH, compile=False)

with open(LABEL_FILE, 'r') as f:
    files = json.load(f)['files']

print("--- PC VISUAL TEST ---")
print("Press 'q' to quit, any other key for next image.")

for entry in files:
    img_path = os.path.join(TEST_IMAGE_DIR, entry['path'])
    if not os.path.exists(img_path): continue

    # 1. Prepare Image
    original_img = cv2.imread(img_path)
    # Ensure it handles the 400x400 correctly for display
    display_img = cv2.resize(original_img, (400, 400))
    
    # Model expects 96x96
    img_input = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    img_input = cv2.resize(img_input, (IMG_SIZE, IMG_SIZE))
    input_data = np.expand_dims(img_input.astype(np.float32) / 255.0, axis=0)

    # 2. Predict
    output = model.predict(input_data, verbose=0)[0] 

    # 3. Scale factors
    scale = 400 / GRID_SIZE # Map 12x12 grid back to 400x400 screen

    # 4. Draw Ground Truth (GREEN)
    bboxes = entry.get('boundingBoxes', [])
    for box in bboxes:
        cv2.rectangle(display_img, 
                     (int(box['x']), int(box['y'])), 
                     (int(box['x']+box['width']), int(box['y']+box['height'])), 
                     (0, 255, 0), 2)

    # 5. Draw Predictions (RED)
    # Get all grid cells with score > 0.5
    detected_indices = np.argwhere(output > THRESHOLD)
    print(f"File: {entry['path']} | Found: {len(detected_indices)}")

    for (y, x, k) in detected_indices:
        score = output[y, x, k]
        # Calculate center
        cx = int(x * scale + scale/2)
        cy = int(y * scale + scale/2)
        
        cv2.circle(display_img, (cx, cy), 4, (0, 0, 255), -1)
        cv2.putText(display_img, f"{score:.2f}", (cx+5, cy), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)

    cv2.imshow("Green: Truth | Red: Prediction", display_img)
    if cv2.waitKey(0) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()