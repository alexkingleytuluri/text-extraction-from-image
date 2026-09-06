import cv2

# Check the 2 and the 5
for digit in ['2', '5']:
    path = f"test_clean/{digit}.png"
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    
    if img is None:
        print(f"Could not read {path}")
        continue
        
    print(f"\n===== WHAT MODEL SEES FOR {digit}.png =====")
    # Print the image as ASCII art (# = white pixel, . = black pixel)
    for row in img:
        line = ""
        for pixel in row:
            if pixel > 127:
                line += "#"
            else:
                line += "."
        print(line)
    print("==========================================\n")