import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split

# 1. Load Data
images = []
labels = []
class_names = sorted(os.listdir("data_clean"))

for label_idx, folder_name in enumerate(class_names):
    folder_path = os.path.join("data_clean", folder_name)
    if not os.path.isdir(folder_path): continue
    for filename in os.listdir(folder_path):
        img = cv2.imread(os.path.join(folder_path, filename), cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        images.append(img / 255.0)
        labels.append(label_idx)

X = np.array(images, dtype=np.float32)
y = np.array(labels, dtype=np.int64)
X = X.reshape(-1, 1, 28, 28)

# Data Augmentation
def augment_data(images, labels):
    aug_images = []
    aug_labels = []
    for img, label in zip(images, labels):
        aug_images.append(img)
        aug_labels.append(label)
        
        M = np.float32([[1, 0, 3], [0, 1, 0]])
        shifted = cv2.warpAffine(img[0], M, (28, 28))
        aug_images.append(shifted.reshape(1, 28, 28))
        aug_labels.append(label)
        
        M = np.float32([[1, 0, -3], [0, 1, 0]])
        shifted = cv2.warpAffine(img[0], M, (28, 28))
        aug_images.append(shifted.reshape(1, 28, 28))
        aug_labels.append(label)
        
        M = cv2.getRotationMatrix2D((14, 14), 15, 1)
        rotated = cv2.warpAffine(img[0], M, (28, 28))
        aug_images.append(rotated.reshape(1, 28, 28))
        aug_labels.append(label)
        
    return np.array(aug_images), np.array(aug_labels)

print(f"Original training images: {len(X)}")
X_aug, y_aug = augment_data(X, y)
print(f"Augmented training images: {len(X_aug)}")

X_train, X_test, y_train, y_test = train_test_split(X_aug, y_aug, test_size=0.2, random_state=14)

train_data = TensorDataset(torch.tensor(X_train), torch.tensor(y_train))
test_data = TensorDataset(torch.tensor(X_test), torch.tensor(y_test))
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
test_loader = DataLoader(test_data, batch_size=32, shuffle=False)

# 2. Define the CNN Model (Added Dropout!)
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
            nn.Dropout(0.5), # NEW: Forces the model to not memorize
            nn.Linear(128, 10)
        )
        
    def forward(self, x):
        return self.network(x)

model = SimpleCNN()
criterion = nn.CrossEntropyLoss()

# NEW: Lower learning rate (0.0001) and added weight_decay to prevent mode collapse
optimizer = optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-4)

# 3. Train the Model
print("Training CNN Model (Tuned Hyperparameters)...")
epochs = 40 # Slightly more epochs since learning rate is lower
for epoch in range(epochs):
    model.train()
    for batch_x, batch_y in train_loader:
        optimizer.zero_grad()
        outputs = model(batch_x)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        
    if (epoch+1) % 5 == 0:
        print(f"Epoch {epoch+1}/{epochs} complete.")

# 4. Evaluate
model.eval()
correct = 0
total = 0
with torch.no_grad():
    for batch_x, batch_y in test_loader:
        outputs = model(batch_x)
        _, predicted = torch.max(outputs.data, 1)
        total += batch_y.size(0)
        correct += (predicted == batch_y).sum().item()

print(f"\nCNN Test Accuracy: {(correct/total)*100:.1f}%")

# 5. Save the model
torch.save({
    'model_state': model.state_dict(),
    'classes': class_names
}, "cnn_model.pth")
print("Model saved as cnn_model.pth")