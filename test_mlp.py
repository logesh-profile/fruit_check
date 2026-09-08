import joblib
import pandas as pd

# Load the trained MLP model
model = joblib.load("mlp_model.pkl")

# New fruit data
new_fruit = pd.DataFrame([{
    "Fruit": "Mango",
    "Temp avg": 28.4,
    "Humidity avg": 80.1,
    "Days remaining": 3.0
}])

# Predict the ripening stage
prediction = model.predict(new_fruit)

print("Predicted Stage:", prediction[0])