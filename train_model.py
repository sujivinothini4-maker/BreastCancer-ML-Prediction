# Breast Cancer Prediction - Model Training

import joblib
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

# 1. Load the dataset
data = load_breast_cancer()

X = data.data
y = data.target

print("=" * 50)
print("BREAST CANCER PREDICTION")
print("=" * 50)

print("Dataset samples:", X.shape[0])
print("Number of features:", X.shape[1])
print("Classes:", data.target_names)

# 2. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Create machine learning models
models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(max_iter=2000))
    ]),
    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        class_weight="balanced"
    )
}

# 4. Train and evaluate models
best_model = None
best_score = -1
best_name = ""

for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=data.target_names
        )
    )

    print("Confusion Matrix:")
    print(confusion_matrix(y_test, predictions))

    if accuracy > best_score:
        best_score = accuracy
        best_model = model
        best_name = name

# 5. Save the trained model
model_data = {
    "model": best_model,
    "feature_names": data.feature_names.tolist(),
    "target_names": data.target_names.tolist(),
    "model_name": best_name
}

joblib.dump(model_data, "breast_cancer_model.pkl")

print("\n" + "=" * 50)
print("MODEL TRAINING COMPLETED!")
print("Selected model:", best_name)
print(f"Test accuracy: {best_score:.4f}")
print("Model saved as breast_cancer_model.pkl")
print("=" * 50)
# Test the model using real dataset samples

print("\n--- SAMPLE PREDICTION TEST ---")

sample_indices = [0, 19, 100, 200]

for index in sample_indices:
    sample = X[index].reshape(1, -1)
    actual = data.target_names[y[index]]
    predicted = data.target_names[best_model.predict(sample)[0]]

    print(f"\nSample index: {index}")
    print(f"Actual class: {actual}")
    print(f"Predicted class: {predicted}")