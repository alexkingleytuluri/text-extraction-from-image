import os
import cv2
import numpy as np
from skimage.feature import hog

def augment_image(img):
    """Creates 3 new variations of the image to help the model learn better"""
    images = [img]
    
    # Shift Right
    M_right = np.float32([[1, 0, 2], [0, 1, 0]])
    images.append(cv2.warpAffine(img, M_right, (28, 28)))
    
    # Shift Left
    M_left = np.float32([[1, 0, -2], [0, 1, 0]])
    images.append(cv2.warpAffine(img, M_left, (28, 28)))
    
    # Rotate 10 degrees
    rows, cols = img.shape
    M_rot = cv2.getRotationMatrix2D((cols/2, rows/2), 10, 1)
    images.append(cv2.warpAffine(img, M_rot, (28, 28)))
    
    return images

def load_dataset(data_folder):
    images = []
    labels = []
    
    for folder_name in os.listdir(data_folder):
        folder_path = os.path.join(data_folder, folder_name)
        if not os.path.isdir(folder_path): continue
            
        # Translate 's-a' back to 'a'
        if folder_name.startswith('s-'):
            label = folder_name.split('-')[1]
        else:
            label = folder_name
            
        for filename in os.listdir(folder_path):
            image_path = os.path.join(folder_path, filename)
            gray = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
            if gray is None: continue
            if gray.shape != (28, 28):
                gray = cv2.resize(gray, (28, 28))
            
            normalized = gray / 255.0
            
            # Generate augmented versions!
            augmented_images = augment_image(normalized)
            
            for aug_img in augmented_images:
                features = hog(aug_img, orientations=9, pixels_per_cell=(4, 4), 
                               cells_per_block=(2, 2), block_norm='L2-Hys')
                images.append(features)
                labels.append(label)
            
    return np.array(images), np.array(labels)

if __name__ == "__main__":
    X, y = load_dataset("data_clean")
    print("Total samples (after augmentation):", len(X))