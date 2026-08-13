from load_dataset import load_dataset
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = load_dataset("data_clean")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# SVM with RBF kernel is much better at shape boundaries than KNN
model = SVC(kernel='rbf', C=5.0, gamma='scale', probability=True)

print("Training SVM model with HOG features...")
model.fit(X_train, y_train)

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"SVM Accuracy: {accuracy * 100:.1f}%")