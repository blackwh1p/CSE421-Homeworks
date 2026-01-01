import serial
import struct
import numpy as np
import os
import time
import random  # Import random library
import scipy.io.wavfile as wav
import scipy.signal as signal

# =======================
# CONFIGURATION
# =======================
SERIAL_PORT = 'COM6'       # Check your port!
BAUD_RATE = 115200
FSDD_PATH = "data/recordings" 
BUFFER_SIZE = 32000 
TARGET_SAMPLE_RATE = 32000 
NUM_CLASSES = 10 

# =======================
# UTILITIES
# =======================
def process_audio_file(filepath):
    """
    Reads wav, resamples to 32kHz, fits to buffer, converts to int16.
    """
    try:
        sr, audio = wav.read(filepath)
    except ValueError:
        return None 

    # Normalize to float [-1, 1]
    if audio.dtype == np.int16:
        audio = audio.astype(np.float32) / 32768.0
    elif audio.dtype == np.uint8:
        audio = (audio.astype(np.float32) - 128.0) / 128.0
    
    # Resample to 32kHz
    if sr != TARGET_SAMPLE_RATE:
        num_samples = int(len(audio) * float(TARGET_SAMPLE_RATE) / sr)
        audio = signal.resample(audio, num_samples)

    # Pad or Truncate
    if len(audio) > BUFFER_SIZE:
        audio = audio[:BUFFER_SIZE]
    else:
        padding = BUFFER_SIZE - len(audio)
        audio = np.concatenate((audio, np.zeros(padding)))

    # Convert to int16
    audio_int16 = (audio * 32767).astype(np.int16)
    return audio_int16

def load_test_data(path, limit=50):
    print(f"Loading files from {path}...")
    files = [f for f in os.listdir(path) if f.endswith('.wav')]
    
    # Filter for 'yweweler' speaker
    test_files = [f for f in files if "yweweler" in f]
    if not test_files: test_files = files

    # --- RANDOMIZE THE LIST ---
    print("Shuffling files...")
    random.shuffle(test_files) 
    # --------------------------

    if limit: test_files = test_files[:limit]
        
    dataset = []
    for fname in test_files:
        data = process_audio_file(os.path.join(path, fname))
        if data is not None:
            # Filename format: digit_speaker_index.wav
            label = int(fname.split('_')[0])
            dataset.append((data, label, fname))
            
    print(f"Loaded {len(dataset)} random samples.")
    return dataset

# =======================
# MAIN LOOP
# =======================
if __name__ == "__main__":
    
    try:
        ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=10)
        print(f"Connected to {SERIAL_PORT}")
        time.sleep(2) 
        ser.reset_input_buffer()
    except Exception as e:
        print(f"Serial Error: {e}")
        exit()

    # Load random data (Limit to 50 samples)
    test_data = load_test_data(FSDD_PATH, limit=50)
    
    correct = 0
    total = 0

    print("-" * 75)
    print(f"{'Filename':<25} | {'True':<5} | {'Pred':<5} | {'Conf':<6} | {'Result'}")
    print("-" * 75)

    for audio_data, true_label, filename in test_data:
        
        # 1. Sync
        board_ready = False
        while not board_ready:
            try:
                line = ser.readline().decode('utf-8', errors='ignore').strip()
                if "WAITING_FOR_AUDIO" in line:
                    board_ready = True
            except: pass
        
        # 2. Send Audio
        ser.write(audio_data.tobytes())
        
        # 3. Receive Prediction
        response = ser.read(40) # 10 floats = 40 bytes
        
        if len(response) != 40:
            print(f"{filename:<25} | {true_label:<5} | ---   | ---    | ❌ Timeout")
            continue

        try:
            probs = struct.unpack('<10f', response)
            pred_class = np.argmax(probs)
            confidence = probs[pred_class]
            
            if pred_class == true_label:
                correct += 1
                status = "✅"
            else:
                status = "❌"
            
            print(f"{filename:<25} | {true_label:<5} | {pred_class:<5} | {confidence:.2f}   | {status}")
            total += 1
            
        except:
            print(f"{filename:<25} | Error unpacking")

    if total > 0:
        print("-" * 75)
        print(f"Final Accuracy: {(correct/total)*100:.2f}%")
    
    ser.close()