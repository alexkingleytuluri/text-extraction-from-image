import os
import cv2
import numpy as np
from skimage.feature import hog

def load_dataset(data_folder):
    images = []
    labels = []
    
    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if not os.path.isdir(folder_path): continue
            
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray is None: continue
            if gray.shape != (28, 28):
                gray = cv2.resize(gray, (28, 28))
            
            # Normalize image
            normalized = gray / 255.0
            
            # EXTRACT HOG FEATURES (Shape fingerprint)
            # HOG captures edges, curves and shape information.
            features = hog(normalized, orientations=9, pixels_per_cell=(4, 4), 
                           cells_per_block=(2, 2), block_norm='L2-Hys')
            
            images.append(features)
            labels.append(folder_name)
            
    return np.array(images), np.array(labels)

if __name__ == "__main__":
    X, y = load_dataset("data_clean")
    print("Total samples:", len(X))
    print("Feature vector size:", X.shape[1]) 