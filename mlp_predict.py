import joblib
import pandas as pd

# Load trained MLP model
model = joblib.load("mlp_model.pkl")

# Get input from user
fruit = input("Enter fruit (Banana/Mango): ")
temperature = float(input("Enter temperature: "))
humidity = float(input("Enter humidity: "))
days_remaining = float(input("Enter days remaining: "))

# Create input data
data = pd.DataFrame([{
    "Fruit": fruit,
    "Temp avg": temperature,
    "Humidity avg": humidity,
    "Days remaining": days_remaining
}])

# Predict
prediction = model.predict(data)[0]

# Display result
print()
print("=" * 40)
print("FRUIT RIPENESS PREDICTION")
print("=" * 40)
print("Fruit              :", fruit)
print("Days remaining     :", days_remaining)
print("Predicted Stage    :", prediction)
print("=" * 40)