import serial
import cv2
import numpy as np
import time
import json
import os

# --- SETTINGS ---
COM_PORT = 'COM6'   
BAUD_RATE = 115200  
TEST_IMAGE_DIR = "dataset/test"
LABEL_FILE = "dataset/test_labels.json"

# --- TUNING (AYARLAR) ---
# Burayı değiştirerek en iyi sonucu bulabilirsin.
# Tavsiye: 0, 10 veya 20 dene. (-128 ile 127 arası)
THRESHOLD = 20       

# --- CONFIG ---
GRID_WIDTH = 12
GRID_HEIGHT = 12
IMG_WIDTH = 96
IMG_HEIGHT = 96
CELL_SIZE = 8 

# --- METRICS ---
total_objects = 0      
total_predictions = 0 
true_positives = 0     
false_positives = 0    
false_negatives = 0    

# --- CONNECTION ---
try:
    ser = serial.Serial(COM_PORT, BAUD_RATE, timeout=10, dsrdtr=True)
except Exception as e:
    print(f"[ERROR] {e}")
    exit()

# --- DATASET ---
with open(LABEL_FILE, 'r') as f:
    ground_truth_list = json.load(f)['files']

def send_image_in_chunks(serial_port, data, chunk_size=32, delay=0.002):
    for i in range(0, len(data), chunk_size):
        serial_port.write(data[i:i+chunk_size])
        time.sleep(delay)

print(f"--- STARTING TEST (Threshold: {THRESHOLD}) ---")
print("!!! PRESS RESET ON BOARD ONCE IF WAITING !!!")

ser.reset_input_buffer()

for i, entry in enumerate(ground_truth_list):
    img_path = os.path.join(TEST_IMAGE_DIR, entry['path'])
    if not os.path.exists(img_path): continue
    
    # Load & Process
    img = cv2.imread(img_path)
    orig_h, orig_w = img.shape[:2]
    scale_x = IMG_WIDTH / orig_w
    scale_y = IMG_HEIGHT / orig_h

    img_resized = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_resized, (IMG_WIDTH, IMG_HEIGHT))
    input_data = np.array(img_resized, dtype=np.float32) - 128.0
    input_data = np.clip(input_data, -128, 127).astype(np.int8)
    
    # Ground Truth
    bboxes = entry.get('boundingBoxes', [])
    total_objects += len(bboxes)
    
    # Track which GT box has been found
    box_already_found = [False] * len(bboxes)

    print(f"\n[{i+1}/{len(ground_truth_list)}] {entry['path']} (Objects: {len(bboxes)})")
    
    # Handshake
    print("   -> Waiting for Handshake '#'...", end='', flush=True)
    handshake = False
    start = time.time()
    while time.time() - start < 10:
        if ser.in_waiting > 0:
            if ser.read(1) == b'#':
                print(" OK!")
                handshake = True
                break
    
    if not handshake: 
        print("\n[TIMEOUT] Handshake failed. Press RESET on board!")
        ser.reset_input_buffer()
        continue

    # Send
    send_image_in_chunks(ser, input_data.tobytes())
    
    # Read Results
    predictions = []
    while True:
        try:
            line = ser.read_until(b'\n').decode('utf-8', errors='ignore').strip()
            if line.startswith("DET:"):
                parts = line.split(':')[1].split(',')
                score = int(parts[2])
                
                # --- THRESHOLD FILTERING ---
                # Karttan gelen veri düşük olsa bile burada eliyoruz
                if score >= THRESHOLD:
                    predictions.append({
                        'x': int(parts[0]),
                        'y': int(parts[1]),
                        'score': score
                    })
            elif line.startswith("END:"):
                break
            elif "Error" in line:
                print(f"   -> BOARD ERROR: {line}")
                break
        except Exception as e:
            break
            
    total_predictions += len(predictions)
    
    # --- EVALUATE PREDICTIONS ---
    if len(predictions) == 0:
        print("   -> No detections (after filtering).")
    
    for pred in predictions:
        cx = (pred['x'] * CELL_SIZE) + (CELL_SIZE // 2)
        cy = (pred['y'] * CELL_SIZE) + (CELL_SIZE // 2)
        
        hit_index = -1 
        
        for idx, box in enumerate(bboxes):
            bx = box['x'] * scale_x
            by = box['y'] * scale_y
            bw = box['width'] * scale_x
            bh = box['height'] * scale_y
            
            if (bx <= cx <= bx + bw) and (by <= cy <= by + bh):
                hit_index = idx
                break
        
        if hit_index != -1:
            if not box_already_found[hit_index]:
                true_positives += 1
                box_already_found[hit_index] = True
                print(f"   -> HIT (New) at Grid({pred['x']},{pred['y']}) Score:{pred['score']}")
            else:
                print(f"   -> HIT (Duplicate) at Grid({pred['x']},{pred['y']}) Score:{pred['score']}")
        else:
            false_positives += 1
            print(f"   -> FALSE ALARM at Grid({pred['x']},{pred['y']}) Score:{pred['score']}")

    missed = box_already_found.count(False)
    false_negatives += missed
    if missed > 0:
        print(f"   -> MISSED {missed} flies.")

# --- FINAL METRICS ---
precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
total_operations = true_positives + false_positives + false_negatives
accuracy = true_positives / total_operations if total_operations > 0 else 0

print("\n" + "="*40)
print(f"Total Ground Truth Objects : {total_objects}")
print(f"Total Predictions Made     : {total_predictions}")
print("-" * 20)
print(f"True Positives (TP)        : {true_positives}")
print(f"False Positives (FP)       : {false_positives}")
print(f"False Negatives (FN)       : {false_negatives}")
print("-" * 20)
print(f"Detection Accuracy         : {accuracy:.2%}")
print(f"Precision                  : {precision:.2%}")
print(f"Recall                     : {recall:.2%}")
print(f"F1 Score                   : {f1:.2f}")
print("="*40)
ser.close()