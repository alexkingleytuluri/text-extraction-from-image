'''import os
import cv2
import numpy as np
from load_dataset import load_dataset
from sklearn.neighbors import KNeighborsClassifier

X, y = load_dataset("data_clean")
X_flat = X.reshape(X.shape[0], -1)

model = KNeighborsClassifier(n_neighbors=3, weights='distance')
model.fit(X_flat, y)

test_folder = "test_clean"
print("Predicting Test Images (KNN K=3)...\n")

files = sorted(os.listdir(test_folder), key=lambda x: (not x[0].isdigit(), x))

for filename in files:
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        
    true_label = os.path.splitext(filename)[0]
    path = os.path.join(test_folder, filename)
    
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None: continue
    if gray.shape != (28, 28):
        gray = cv2.resize(gray, (28, 28))
        
    normalized = gray / 255.0
    flat = normalized.reshape(1, -1)
    
    prediction = model.predict(flat)[0]
    print(f"File: {filename} | True: {true_label} | Predicted: {prediction}")'''

import os
import cv2
import numpy as np
from load_dataset import load_dataset
from sklearn.svm import SVC
from skimage.feature import hog

# Train on ALL data
X, y = load_dataset("data_clean")
model = SVC(kernel='rbf', C=5.0, gamma='scale', probability=True)
model.fit(X, y)

test_folder = "test_clean"
print("Predicting Test Images (SVM + HOG)...\n")

files = sorted(os.listdir(test_folder), key=lambda x: (not x[0].isdigit(), x))

for filename in files:
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        
    true_label = os.path.splitext(filename)[0]
    path = os.path.join(test_folder, filename)
    
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None: continue
    if gray.shape != (28, 28):
        gray = cv2.resize(gray, (28, 28))
        
    normalized = gray / 255.0
    
    # EXTRACT HOG FEATURES for the test image
    features = hog(normalized, orientations=9, pixels_per_cell=(4, 4), 
                   cells_per_block=(2, 2), block_norm='L2-Hys')
    flat = features.reshape(1, -1)
    
    prediction = model.predict(flat)[0]
    print(f"File: {filename} | True: {true_label} | Predicted: {prediction}")