from load_dataset import load_dataset
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

X, y = load_dataset("data_clean")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("Finding the best mathematical parameters for SVM...")
param_grid = {
    'C': [1, 5, 10, 50, 100],
    'gamma': ['scale', 0.01, 0.1, 1, 10]
}

grid = GridSearchCV(SVC(kernel='rbf'), param_grid, cv=3, n_jobs=-1)
grid.fit(X_train, y_train)

print(f"Best parameters found: {grid.best_params_}")
model = grid.best_estimator_

predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print(f"Tuned SVM Accuracy: {accuracy * 100:.1f}%")