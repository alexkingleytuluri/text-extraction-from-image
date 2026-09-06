import cv2
import numpy as np

# Let's check Test image 1
img = cv2.imread("test/1.png", cv2.IMREAD_GRAYSCALE)

if img is None:
    print("ERROR: Could not read test/1.png")
else:
    # Count how many white pixels (the digit) there are
    white_pixels = np.sum(img > 127)
    total_pixels = img.size
    percentage = (white_pixels / total_pixels) * 100
    
    print(f"Image shape: {img.shape}")
    print(f"White pixel percentage: {percentage:.2f}%")
    
    if percentage < 5:
        print("WARNING: The image is almost completely black! The digit is too small.")
    elif percentage > 60:
        print("WARNING: The image is almost completely white! There is too much noise/background.")
    else:
        print("Digit size looks normal.")
        
    # Show the image on your screen
    # We scale it up 10x so you can actually see it
    big_img = cv2.resize(img, (280, 280), interpolation=cv2.INTER_NEAREST)
    cv2.imshow("What the model sees for test/1.png", big_img)
    print("Close the image window to continue...")
    cv2.waitKey(0)
    cv2.destroyAllWindows()