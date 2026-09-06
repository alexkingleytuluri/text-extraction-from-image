from load_dataset import load_dataset
from sklearn.svm import SVC
import joblib


print("=" * 70)
print("FINAL HANDWRITING OCR TRAINING")
print("=" * 70)

print("\nLoading training data...")

X, y = load_dataset("data_clean")

print(f"Training samples : {len(X)}")
print(f"Feature size     : {X.shape[1]}")
print(f"Classes          : {len(set(y))}")

print("\nTraining final SVM...")

model = SVC(
    kernel="rbf",
    C=50.0,
    gamma=0.01
)

model.fit(X, y)

print("Training complete.")

# Save the finalized model
model_path = "handwriting_svm.joblib"
joblib.dump(model, model_path)

print(f"\nModel saved to   : {model_path}")

print("\n" + "=" * 70)
print("FINAL MODEL CONFIGURATION")
print("=" * 70)

print("HOG orientations : 9")
print("Pixels per cell  : (4, 4)")
print("Cells per block  : (2, 2)")
print("SVM kernel       : RBF")
print("SVM C            : 50.0")
print("SVM gamma        : 0.01")
print(f"Training samples : {len(X)}")
print(f"Feature size     : {X.shape[1]}")

print("\nFinal model ready for prediction.")