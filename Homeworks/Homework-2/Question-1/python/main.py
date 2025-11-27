import os.path as osp
from data_utils import read_data
from feature_utils import create_features
from sklearn import metrics
import sklearn2c
from matplotlib import pyplot as plt

# --- CHANGE: Set to power of 2 for STM32 FFT compatibility ---
TIME_PERIODS = 64
STEP_DISTANCE = 32
# -------------------------------------------------------------

DATA_PATH = osp.join("WISDM_ar_v1.1", "WISDM_ar_v1.1_raw.txt")

# Read and Filter Data
data_df = read_data(DATA_PATH)
df_train = data_df[data_df["user"] <= 28]
#df_test = data_df[data_df["user"] > 28]

# Extract Features
train_segments_df, train_labels = create_features(df_train, TIME_PERIODS, STEP_DISTANCE)
#test_segments_df, test_labels = create_features(df_test, TIME_PERIODS, STEP_DISTANCE)

# Train Bayes Classifier
bayes = sklearn2c.BayesClassifier()
bayes.train(train_segments_df, train_labels)

# Evaluate
#bayes_preds = bayes.predict(test_segments_df)
#conf_matrix = metrics.confusion_matrix(test_labels, bayes_preds)
#print(f"Accuracy: {metrics.accuracy_score(test_labels, bayes_preds)}")

# Export for Microcontroller
# This generates 'bayes_cls_config.h' and 'bayes_har_config.c'
bayes.export("../bayes_cls_config")
print("Model exported to bayes_cls_config.h/.c")