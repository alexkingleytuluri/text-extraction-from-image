import cv2
import os

# Check the first image in data/0
path = os.path.join("data", "0", "1.png")
if not os.path.exists(path):
    print(f"ERROR: {path} does not exist. Put images in the data folder!")
else:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print("ERROR: Could not read the image.")
    else:
        print(f"Image shape: {img.shape}")
        print(f"Max pixel value: {img.max()} (Should be 255 for a raw photo)")
        print(f"Min pixel value: {img.min()} (Should be close to 0 for a raw photo)")
        
        if img.shape == (28, 28):
            print("\n🚨 WARNING: This image is 28x28! This means it is ALREADY CORRUPTED.")
            print("You MUST take brand new photos with your phone and replace them.")
        else:
            print("\n✅ GOOD: This is a raw photo. You can proceed with preprocessing!")