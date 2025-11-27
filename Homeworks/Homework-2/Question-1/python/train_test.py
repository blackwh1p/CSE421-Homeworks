import os.path as osp
import numpy as np
from sklearn import metrics
import sklearn2c
import matplotlib.pyplot as plt
from data_utils import read_data
from feature_utils import create_features

# --- CONFIGURATION (Must match your MCU settings) ---
TIME_PERIODS = 64
STEP_DISTANCE = 32
# ----------------------------------------------------

def main():
    # 1. Load Data
    print("1. Loading Dataset...")
    DATA_PATH = osp.join("WISDM_ar_v1.1", "WISDM_ar_v1.1_raw.txt")
    data_df = read_data(DATA_PATH)

    # 2. Split Data (Same split as before)
    print("2. Splitting Train/Test...")
    df_train = data_df[data_df["user"] <= 28]
    df_test = data_df[data_df["user"] > 28]

    # 3. Feature Extraction
    print("3. Extracting Features (this takes a few seconds)...")
    train_X, train_y = create_features(df_train, TIME_PERIODS, STEP_DISTANCE)
    test_X, test_y = create_features(df_test, TIME_PERIODS, STEP_DISTANCE)

    print(f"   Training Samples: {len(train_X)}")
    print(f"   Testing Samples:  {len(test_X)}")

    # 4. Train Model
    print("4. Training Bayes Classifier...")
    # case=3 usually maps to using separate means/variances per class
    bayes = sklearn2c.BayesClassifier(case=3) 
    bayes.train(train_X, train_y)

    # 5. Test on Python (PC)
    print("5. Testing on Python...")
    
    # 'predict' returns probabilities/likelihoods [N_samples, N_classes]
    raw_predictions = bayes.predict(test_X)
    
    # Convert probabilities to Class Index (0, 1, 2...)
    predicted_indices = np.argmax(raw_predictions, axis=1)
    
    # Convert Indices back to String Labels (e.g., 0 -> "Walking")
    # sklearn2c stores class names in alphabetical order usually
    class_names = sorted(np.unique(train_y)) 
    predicted_labels = [class_names[i] for i in predicted_indices]

    # 6. Calculate Accuracy
    acc = metrics.accuracy_score(test_y, predicted_labels)
    print("\n" + "="*40)
    print(f"PYTHON BASELINE ACCURACY: {acc * 100:.2f}%")
    print("="*40 + "\n")

    # 7. Confusion Matrix
    print("Generating Confusion Matrix...")
    cm = metrics.confusion_matrix(test_y, predicted_labels, labels=class_names)
    disp = metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 8))
    disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
    plt.title(f"Python Accuracy: {acc*100:.2f}%")
    plt.tight_layout()
    plt.show()

    # 8. Export (Optional, just to confirm)
    # bayes.export("bayes_cls_config")

if __name__ == "__main__":
    main()