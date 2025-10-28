import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import classification_report
import joblib  # to save the trained model

CSV_PATH = "../GestureData/gesture_data_full.csv"
MODEL_PATH = "../GestureData/gesture_classifier.joblib"
ENCODER_PATH = "../GestureData/label_encoder.joblib"

# 1. load data
Xs = []
ys = []

with open(CSV_PATH, 'r') as f:
    reader = csv.reader(f)
    next(reader)  # skip first row
    for row in reader:
        *features, label = row
        Xs.append([float(v) for v in features])
        ys.append(label)

X = np.array(Xs)           # shape (N, 63)
y_text = np.array(ys)      # shape (N,)

# 2. turn string labels ("Thumb_Left", etc.) into integers (0,1,2,...)
le = LabelEncoder()
y = le.fit_transform(y_text)

# 3. train/val split
X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

# 4. define model
clf = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation='relu',
    max_iter=500
)

# 5. train
clf.fit(X_train, y_train)

# 6. evaluate
y_pred = clf.predict(X_val)
print(classification_report(y_val, y_pred, target_names=le.classes_))

# 7. save model + encoder for runtime use
joblib.dump(clf, MODEL_PATH)
joblib.dump(le, ENCODER_PATH)
print(f"Saved model to {MODEL_PATH}")
print(f"Saved label encoder to {ENCODER_PATH}")
