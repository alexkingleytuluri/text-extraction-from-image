import cv2
import numpy as np
import os
from load_dataset import load_dataset
from sklearn.svm import SVC
from skimage.feature import hog
import textdistance

# 1. Train the SVM Model
print("Training SVM model on augmented data...")
X, y = load_dataset("data_clean")
model = SVC(kernel='rbf', C=50, gamma=0.01)
model.fit(X, y)

# 2. Helper function to process a single cropped letter
def process_single_char(crop):
    max_dim = max(crop.shape)
    if max_dim == 0: return None
    scale = 20.0 / max_dim
    new_w, new_h = int(crop.shape[1] * scale), int(crop.shape[0] * scale)
    resized = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    canvas = np.zeros((28, 28), dtype=np.uint8)
    x_off = (28 - new_w) // 2
    y_off = (28 - new_h) // 2
    canvas[y_off:y_off+new_h, x_off:x_off+new_w] = resized
    
    cy, cx = np.where(canvas > 0)
    if len(cx) > 0 and len(cy) > 0:
        shift_x = int(np.round(14 - np.mean(cx)))
        shift_y = int(np.round(14 - np.mean(cy)))
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        canvas = cv2.warpAffine(canvas, M, (28, 28))
    return canvas

# 3. Medical Dictionary for Spell Correction
DICTIONARY = ["Dolo", "650", "Crocin", "Aspirin", "Azithromycin", "Paracetamol", "Ibuprofen"]

def correct_spelling(word):
    best_match = word
    lowest_distance = 999
    
    for dict_word in DICTIONARY:
        dist = textdistance.levenshtein(word.lower(), dict_word.lower())
        if dist < lowest_distance:
            lowest_distance = dist
            best_match = dict_word
            
    max_allowed_dist = len(word) * 0.3 
    if lowest_distance <= max_allowed_dist:
        return best_match
    return word

# 4. Load the word image
image_path = "word_img.jpeg"
if not os.path.exists(image_path): image_path = "word_img.jpg"
if not os.path.exists(image_path): image_path = "word_img.png"

img = cv2.imread(image_path)
if img is None:
    print(f"ERROR: Could not read {image_path}.")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 5. Threshold and find letters
_, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
valid_contours = [c for c in contours if cv2.contourArea(c) > 100]

valid_contours = sorted(valid_contours, key=lambda c: cv2.boundingRect(c)[1])

lines = []
current_line = []
last_y = -9999

for c in valid_contours:
    x, y, w, h = cv2.boundingRect(c)
    if abs(y - last_y) > 30: 
        if current_line: lines.append(current_line)
        current_line = [c]
    else:
        current_line.append(c)
    last_y = y

if current_line: lines.append(current_line)

print(f"\nFound {len(lines)} lines of text. Predicting...\n")

# 6. Process and predict each line
for i, line in enumerate(lines):
    line = sorted(line, key=lambda c: cv2.boundingRect(c)[0])
    
    raw_predicted_word = ""
    for c in line:
        x, y, w, h = cv2.boundingRect(c)
        crop = binary[y:y+h, x:x+w]
        
        processed = process_single_char(crop)
        if processed is None: continue
        
        normalized = processed / 255.0
        features = hog(normalized, orientations=9, pixels_per_cell=(4, 4), 
                       cells_per_block=(2, 2), block_norm='L2-Hys')
        flat = features.reshape(1, -1)
        
        prediction = model.predict(flat)[0]
        raw_predicted_word += str(prediction)
        
    corrected_word = correct_spelling(raw_predicted_word)
    print(f"Line {i+1}: {raw_predicted_word}  -->  Corrected: {corrected_word}")