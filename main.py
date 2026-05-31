import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# Create folders
os.makedirs("data", exist_ok=True)
os.makedirs("charts", exist_ok=True)
os.makedirs("models", exist_ok=True)

# Create realistic sample real estate dataset
np.random.seed(42)

num_houses = 300

area_sqft = np.random.randint(600, 4500, num_houses)
bedrooms = np.random.randint(1, 7, num_houses)
bathrooms = np.random.randint(1, 5, num_houses)
house_age = np.random.randint(0, 50, num_houses)
distance_to_city = np.random.uniform(1, 40, num_houses)
location_score = np.random.randint(1, 11, num_houses)

# Price formula with realistic noise
price = (
    area_sqft * 120
    + bedrooms * 15000
    + bathrooms * 12000
    - house_age * 1500
    - distance_to_city * 2500
    + location_score * 20000
    + np.random.normal(0, 30000, num_houses)
)

price = np.maximum(price, 50000)

df = pd.DataFrame({
    "Area_Sqft": area_sqft,
    "Bedrooms": bedrooms,
    "Bathrooms": bathrooms,
    "House_Age": house_age,
    "Distance_To_City": distance_to_city.round(2),
    "Location_Score": location_score,
    "Price": price.round(2)
})

df.to_csv("data/house_prices.csv", index=False)

print("Dataset created successfully!")
print("\nFirst 5 rows:")
print(df.head())

print("\nDataset summary:")
print(df.describe())

# Features and target
X = df[[
    "Area_Sqft",
    "Bedrooms",
    "Bathrooms",
    "House_Age",
    "Distance_To_City",
    "Location_Score"
]]

y = df["Price"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation:")
print(f"Mean Absolute Error: {mae:.2f}")
print(f"R2 Score: {r2:.2f}")

# Save model
joblib.dump(model, "models/house_price_model.pkl")
print("\nModel saved successfully inside models folder.")

# Chart 1: Area vs Price
plt.figure()
plt.scatter(df["Area_Sqft"], df["Price"])
plt.title("House Area vs Price")
plt.xlabel("Area in Square Feet")
plt.ylabel("Price")
plt.tight_layout()
plt.savefig("charts/area_vs_price.png")
plt.close()

# Chart 2: Actual vs Predicted Prices
plt.figure()
plt.scatter(y_test, y_pred)
plt.title("Actual vs Predicted House Prices")
plt.xlabel("Actual Price")
plt.ylabel("Predicted Price")
plt.tight_layout()
plt.savefig("charts/actual_vs_predicted.png")
plt.close()

# Chart 3: Average price by location score
location_avg = df.groupby("Location_Score")["Price"].mean()

plt.figure()
location_avg.plot(kind="bar")
plt.title("Average Price by Location Score")
plt.xlabel("Location Score")
plt.ylabel("Average Price")
plt.tight_layout()
plt.savefig("charts/price_by_location_score.png")
plt.close()

print("Charts saved successfully inside charts folder.")