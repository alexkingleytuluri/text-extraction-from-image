import os
import cv2
import numpy as np

def process_image(img):
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
        
    _, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours: return None
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    cropped = binary[y:y+h, x:x+w]
    
    max_dim = max(w, h)
    if max_dim == 0: return None
    scale = 20.0 / max_dim
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(cropped, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    return canvas

def process_folder(src, dst):
    if not os.path.exists(dst): os.makedirs(dst)
    for filename in os.listdir(src):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            img = cv2.imread(os.path.join(src, filename))
            if img is not None:
                processed = process_image(img)
                if processed is not None:
                    cv2.imwrite(os.path.join(dst, filename), processed)

if __name__ == "__main__":
    for folder_name in os.listdir("data"):
        src_folder = os.path.join("data", folder_name)
        if os.path.isdir(src_folder):
            process_folder(src_folder, os.path.join("data_clean", folder_name))
            print(f"Processed data/{folder_name}")
    process_folder("test", "test_clean")
    print("Processed test")