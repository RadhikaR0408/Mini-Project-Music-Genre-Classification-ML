import numpy as np
import librosa
import pickle

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report

DATASET_PATH = "genres_original"
MODEL_PATH = "genre_model.pkl"

X = []
y = []

# -----------------------------
# Feature Extraction
# -----------------------------
def extract_features(file_path):
    y_audio, sr = librosa.load(file_path, duration=30)
    mfcc = librosa.feature.mfcc(y=y_audio, sr=sr, n_mfcc=13)

    mfcc_mean = np.mean(mfcc, axis=1)
    mfcc_std = np.std(mfcc, axis=1)

    return np.hstack((mfcc_mean, mfcc_std))


# -----------------------------
# Load Dataset
# -----------------------------
label_map = {}
label_id = 0

for genre in sorted(os.listdir(DATASET_PATH)):
    genre_path = os.path.join(DATASET_PATH, genre)

    if not os.path.isdir(genre_path):
        continue

    label_map[label_id] = genre

    for file in os.listdir(genre_path):
        file_path = os.path.join(genre_path, file)
        features = extract_features(file_path)

        X.append(features)
        y.append(label_id)

    label_id += 1


X = np.array(X)
y = np.array(y)

# -----------------------------
# Train / Test Split
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# -----------------------------
# Train Model
# -----------------------------
model = KNeighborsClassifier(n_neighbors=5)
model.fit(X_train, y_train)

# -----------------------------
# Evaluation
# -----------------------------
y_pred = model.predict(X_test)

print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))

# -----------------------------
# Save Model
# -----------------------------
with open(MODEL_PATH, "wb") as f:
    pickle.dump((model, label_map), f)
