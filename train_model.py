# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load dataset
df = pd.read_csv("heart.csv")

# Prepare features and target
X = df.drop("target", axis=1)
y = df["target"]

# Train the RandomForest model with basic tuning
model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
model.fit(X, y)

# Save the trained model
joblib.dump(model, "heart_model.pkl")

print("✅ Model trained and saved as 'heart_model.pkl'")
