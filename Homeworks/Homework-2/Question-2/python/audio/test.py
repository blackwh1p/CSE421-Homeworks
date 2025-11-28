import os
import time
import struct
import numpy as np
import serial
import scipy.signal as sig

from mfcc_func import create_mfcc_features  # senin modülün

# ==== KLASÖR AYARLARI ====
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
FSDD_PATH = os.path.join(PROJECT_ROOT, "recordings")  # wav dosyaların burada

# ==== MFCC PARAMETRELERİ (STM32 tarafıyla aynı) ====
FFTSize = 1024
sample_rate = 8000
numOfMelFilters = 20
numOfDctOutputs = 13     # nbDctOutputs
window = sig.get_window("hamming", FFTSize)

# ==== VERİYİ YÜKLE (önceki kodunla aynı mantık) ====
recordings_list = [os.path.join(FSDD_PATH, rec) for rec in os.listdir(FSDD_PATH)]
test_list  = {rec for rec in recordings_list if "yweweler" in os.path.basename(rec)}
train_list = set(recordings_list) - test_list

_, _ = create_mfcc_features(train_list, FFTSize, sample_rate,
                            numOfMelFilters, numOfDctOutputs, window)
test_mfcc_features, test_labels = create_mfcc_features(test_list, FFTSize,
                                                       sample_rate,
                                                       numOfMelFilters,
                                                       numOfDctOutputs,
                                                       window)

test_mfcc_features = np.asarray(test_mfcc_features, dtype=np.float32)
test_labels = np.asarray(test_labels, dtype=np.int32)

num_features = numOfDctOutputs * 2   # STM32 tarafıyla aynı

# ==== SERIAL BAĞLANTI ====
# COM portu kendine göre değiştir (Windows: "COM5", Linux: "/dev/ttyACM0")
ser = serial.Serial(
    port="COM6",
    baudrate=115200,
    bytesize=serial.EIGHTBITS,
    parity=serial.PARITY_NONE,
    stopbits=serial.STOPBITS_ONE,
    timeout=1,
)

time.sleep(2.0)  # board reset için küçük bekleme

NUM_CLASSES = 10  # senin problemine göre ayarla (örn. 10 digit)

correct = 0
total = 0

for feats, label in zip(test_mfcc_features, test_labels):
    # features boyutu kontrol
    if feats.size != num_features:
        print("Feature length mismatch:", feats.size)
        continue

    # ---- FEATURES → BOARD ----
    ser.write(feats.astype("<f4").tobytes())   # little-endian float32

    # ---- BOARD → PREDICTION ----
    bytes_expected = NUM_CLASSES * 4  # int32
    rx = b""
    while len(rx) < bytes_expected:
        chunk = ser.read(bytes_expected - len(rx))
        if not chunk:
            break
        rx += chunk

    if len(rx) != bytes_expected:
        print("Timeout / incomplete read from board")
        continue

    # int32 prediction vector
    preds = struct.unpack("<" + "i" * NUM_CLASSES, rx)
    preds = np.asarray(preds, dtype=np.int32)

    pred_class = int(np.argmax(preds))
    total += 1
    if pred_class == int(label):
        correct += 1

    print(f"True: {label}, Pred: {pred_class}, Raw: {preds}")

ser.close()

if total > 0:
    acc = correct * 100 / total
    print(f"\nAccuracy over UART/STM32: {correct} * {100} / {total} = %{acc:.3f}")
else:
    print("No valid samples tested.")
