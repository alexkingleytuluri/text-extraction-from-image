import os
import cv2
import numpy as np

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
            normalized = gray / 255.0
            images.append(normalized)
            labels.append(folder_name)
    return np.array(images), np.array(labels)

if __name__ == "__main__":
    X, y = load_dataset("data_clean")
    print("Total samples:", len(X))