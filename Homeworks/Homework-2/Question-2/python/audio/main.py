import os
import scipy.signal as sig
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import sklearn2c
from matplotlib import pyplot as plt

# -----------------------------
# DIRECTORY FIXES BASED ON YOUR FOLDER STRUCTURE
# -----------------------------

# Project root directory (folder where this script is located)
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))

# Your audio dataset folder
FSDD_PATH = os.path.join(PROJECT_ROOT, "recordings")

# MFCC function folder
from mfcc_func import create_mfcc_features

# Output folders (created automatically)
CLASSIFICATION_MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
CLASSIFICATION_EXPORT_DIR = os.path.join(PROJECT_ROOT, "export")

os.makedirs(CLASSIFICATION_MODEL_DIR, exist_ok=True)
os.makedirs(CLASSIFICATION_EXPORT_DIR, exist_ok=True)

# -----------------------------
# ORIGINAL CODE (UNCHANGED)
# -----------------------------

model_save_path = os.path.join(CLASSIFICATION_MODEL_DIR, "kws_knn.joblib")
export_path = os.path.join(CLASSIFICATION_EXPORT_DIR, "kws_knn_config")

recordings_list = [os.path.join(FSDD_PATH, rec_path) for rec_path in os.listdir(FSDD_PATH)]

FFTSize = 1024
sample_rate = 8000
numOfMelFilters = 20
numOfDctOutputs = 13
window = sig.get_window("hamming", FFTSize)

test_list = {record for record in recordings_list if "yweweler" in os.path.basename(record)}
train_list = set(recordings_list) - test_list

train_mfcc_features, train_labels = create_mfcc_features(
    train_list, FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs, window
)

test_mfcc_features, test_labels = create_mfcc_features(
    test_list, FFTSize, sample_rate, numOfMelFilters, numOfDctOutputs, window
)

knn = sklearn2c.KNNClassifier(n_neighbors=3)
knn.train(train_mfcc_features, train_labels, model_save_path)

knn_preds = knn.predict(test_mfcc_features)
conf_matrix = confusion_matrix(test_labels, knn_preds.argmax(axis=1))
cm_display = ConfusionMatrixDisplay(confusion_matrix=conf_matrix)
cm_display.plot()
cm_display.ax_.set_title("KNN Classifier Confusion Matrix")
plt.show()

knn.export(export_path)
