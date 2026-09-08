import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report


# 1. Load dataset
data = pd.read_csv("dataset/tamilnadu_fruit_ripening_corrected.csv")

print("Dataset loaded successfully!")
print("Shape:", data.shape)
print(data.head())


# 2. Input features and target
# We only use Fruit and Days remaining
X = data[["Fruit", "Days remaining"]]
y = data["Stage"]


# 3. Separate categorical and numerical columns
categorical_features = ["Fruit"]
numerical_features = ["Days remaining"]


# 4. Preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        (
            "cat",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical_features
        ),
        (
            "num",
            StandardScaler(),
            numerical_features
        )
    ]
)


# 5. Create MLP model
model = MLPClassifier(
    hidden_layer_sizes=(64, 32),
    activation="relu",
    solver="adam",
    max_iter=500,
    random_state=42
)


# 6. Create complete pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# 7. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\nTraining MLP model...")


# 8. Train model
pipeline.fit(X_train, y_train)


print("Training completed!")


# 9. Make predictions
y_pred = pipeline.predict(X_test)


# 10. Accuracy
accuracy = accuracy_score(y_test, y_pred)

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))


# 11. Save trained model
joblib.dump(pipeline, "mlp_model.pkl")

print("\nMLP model saved as: mlp_model.pkl")