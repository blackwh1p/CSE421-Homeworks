import serial
import cv2
import numpy as np
import time
import os
import random
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

# --- SETTINGS ---
COM_PORT = 'COM6'        # Check Device Manager
BAUD_RATE = 115200       
TEST_DIR = "dataset/test" 
IMG_SIZE = 96            

# --- CLASS NAMES (16 Classes for Tomato) ---
CLASS_NAMES = [
    "alternaria_d", "alternaria_mite_d", "bacterial_floundering_d",
    "blossom_end_rot_d", "caterpillars_p", "fusarium_d",
    "healthy_fruit", "healthy_leaf", "helicoverpa_armigera_p",
    "mite_d", "nitrogen_exces_d", "spodoptera_frugiperda_p",
    "sunburn_d", "tomato_late_blight_d", "tuta_absoluta_p", "virosis_d"
]

# --- CONNECTION ---
print(f"[INIT] Connecting to {COM_PORT}...")
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=10, dsrdtr=True)
    print("[SUCCESS] Port opened.")
except Exception as e:
    print(f"[ERROR] {e}")
    exit()

# --- LOAD TEST IMAGES ---
test_images = []
for label_name in os.listdir(TEST_DIR):
    folder_path = os.path.join(TEST_DIR, label_name)
    if os.path.isdir(folder_path) and label_name in CLASS_NAMES:
        true_index = CLASS_NAMES.index(label_name)
        for file in os.listdir(folder_path):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                test_images.append({
                    'path': os.path.join(folder_path, file),
                    'label': label_name,
                    'index': true_index
                })

random.shuffle(test_images)
# Limit to 100 images for a manageable test run. Remove [:100] to test ALL images.
test_images = test_images[:100] 

# --- METRIC TRACKERS ---
y_true = []
y_pred = []

# --- HELPER ---
def send_image_in_chunks(serial_port, data, chunk_size=32, delay=0.002):
    for i in range(0, len(data), chunk_size):
        serial_port.write(data[i:i+chunk_size])
        time.sleep(delay)

print(f"\n--- STARTING MCU VISUAL TEST ({len(test_images)} IMAGES) ---")
print("Press 'q' in the image window to stop early, any other key for next image.")
ser.reset_input_buffer()

for i, entry in enumerate(test_images):
    img = cv2.imread(entry['path'])
    if img is None: continue
    
    # Preprocessing for MCU
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    input_data = np.array(img_resized, dtype=np.float32) - 128.0
    input_data = np.clip(input_data, -128, 127).astype(np.int8)
    
    # Handshake
    handshake = False
    start_time = time.time()
    while time.time() - start_time < 5:
        if ser.in_waiting > 0 and ser.read(1) == b'#':
            handshake = True
            break
    
    if not handshake:
        print("\n[TIMEOUT] Handshake failed. Skipping...")
        ser.reset_input_buffer()
        continue

    # Send Image
    send_image_in_chunks(ser, input_data.tobytes())
    
    # Read Result
    try:
        line = ser.read_until(b'\n').decode('utf-8', errors='ignore').strip()
        if line.startswith("CLASS:"):
            parts = line.split(':')[1].split(',')
            pred_index = int(parts[0])
            pred_score = int(parts[1])
            
            pred_name = CLASS_NAMES[pred_index] if 0 <= pred_index < len(CLASS_NAMES) else "Unknown"
            is_correct = (pred_index == entry['index'])
            
            # Store results for final report
            y_true.append(entry['index'])
            y_pred.append(pred_index)
            
            # --- VISUALIZATION FOR REPORT ---
            display_img = cv2.resize(img, (500, 500))
            
            # Text configuration
            true_text = f"True: {entry['label']}"
            pred_text = f"Pred: {pred_name} (Score: {pred_score})"
            color = (0, 255, 0) if is_correct else (0, 0, 255) # BGR: Green if correct, Red if wrong
            
            cv2.putText(display_img, true_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(display_img, pred_text, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            print(f"[{i+1}/{len(test_images)}] {true_text} | {pred_text}")
            
            cv2.imshow("MCU Classification Test", display_img)
            key = cv2.waitKey(0) & 0xFF
            if key == ord('q'):
                break
            
        else:
            print(f"[{i+1}] Invalid response: {line}")
    except Exception as e:
        print(f"Error: {e}")

cv2.destroyAllWindows()
ser.close()

# ==========================================
# GENERATE REPORT DATA & PLOTS
# ==========================================
if len(y_true) > 0:
    print("\n" + "="*50)
    print("FINAL CLASSIFICATION REPORT FOR THE REPORT PDF")
    print("="*50)
    
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    rec = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)

    print(f"Overall Accuracy : {acc:.4f}")
    print(f"Weighted Precision : {prec:.4f}")
    print(f"Weighted Recall    : {rec:.4f}")
    print(f"Weighted F1-Score  : {f1:.4f}")
    print("-" * 50)
    
    # Generate detailed text report
    report = classification_report(y_true, y_pred, target_names=CLASS_NAMES, labels=range(len(CLASS_NAMES)), zero_division=0)
    print(report)

    # --- CONFUSION MATRIX PLOT ---
    cm = confusion_matrix(y_true, y_pred, labels=range(len(CLASS_NAMES)))
    
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('MCU Classification Confusion Matrix')
    plt.ylabel('Actual Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    
    # Save the plot for your report
    plt.savefig('confusion_matrix.png', dpi=300)
    print("[SUCCESS] 'confusion_matrix.png' saved successfully for your report.")
    
    # Show the plot
    plt.show()

else:
    print("No predictions were made.")