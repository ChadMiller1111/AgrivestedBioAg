# AgrivestedBioAg
Agrivested Ai testing for soil samples from lab reports to give advice to farmers 
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
import joblib

# Define sample columns (soil nutrient parameters)
columns = [
    "pH", "Organic_Matter", "Nitrogen", "Phosphorus", "Potassium", "Calcium",
    "Magnesium", "Sulfur", "Zinc", "Iron", "Manganese", "Copper", "Boron",
    "CEC", "Base_Saturation"
]

# Generate synthetic training data (replace with real soil lab data)
np.random.seed(42)
data = pd.DataFrame(np.random.rand(500, len(columns)) * 100, columns=columns)

# Target variable (e.g., soil health score based on historical data)
data["Soil_Health_Score"] = np.random.rand(500) * 100

# Splitting data into training and testing sets
X = data[columns]
y = data["Soil_Health_Score"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train a RandomForest model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train_scaled, y_train)

# Save the trained model and scaler
joblib.dump(model, "soil_nutrient_ai_model.pkl")
joblib.dump(scaler, "soil_nutrient_scaler.pkl")

print("Model and scaler saved successfully!")
