import os
import cv2
import numpy as np
import torch
import torch.nn as nn

# 1. Redefine Model Architecture (MUST exactly match train_cnn.py)
class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 7 * 7, 128),
            nn.ReLU(),
            nn.Dropout(0.5), # Added this missing layer!
            nn.Linear(128, 10)
        )
    def forward(self, x):
        return self.network(x)

# 2. Load the trained model
checkpoint = torch.load("cnn_model.pth")
model = SimpleCNN()
model.load_state_dict(checkpoint['model_state'])
model.eval()
class_names = checkpoint['classes']

test_folder = "test_clean"
print("Predicting Test Images with CNN...\n")

files = sorted(os.listdir(test_folder), key=lambda x: int(x.split('.')[0]) if x.split('.')[0].isdigit() else 999)

for filename in files:
    if not filename.lower().endswith(('.png', '.jpg', '.jpeg')): continue
        
    true_label = os.path.splitext(filename)[0]
    path = os.path.join(test_folder, filename)
    
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: continue
    if img.shape != (28, 28):
        img = cv2.resize(img, (28, 28))
        
    # Preprocess for CNN
    img = img / 255.0
    img = img.reshape(1, 1, 28, 28)
    tensor_img = torch.tensor(img, dtype=torch.float32)
    
    # Predict
    with torch.no_grad():
        output = model(tensor_img)
        _, predicted = torch.max(output, 1)
        prediction = class_names[predicted.item()]
        
    print(f"File: {filename} | True: {true_label} | Predicted: {prediction}")