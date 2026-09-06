import cv2
import os
import numpy as np

# Let's check data_clean/0/1.png and data_clean/1/1.png
for digit in ['0', '1', '2']:
    path = f"data_clean/{digit}/1.png"
    if not os.path.exists(path):
        print(f"File not found: {path}")
        continue
        
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"\n=== {path} ===\nCould not read image!")
        continue
        
    print(f"\n===== WHAT MODEL SEES FOR data_clean/{digit}/1.png =====")
    print(f"Shape: {img.shape} | White pixel %: {(np.sum(img > 127)/img.size)*100:.1f}%")
    for row in img:
        line = ""
        for pixel in row:
            if pixel > 127:
                line += "#"
            else:
                line += "."
        print(line)
    print("==========================================\n")