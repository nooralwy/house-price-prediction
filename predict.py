import joblib
import pandas as pd

model = joblib.load("models/house_price_model.pkl")

new_house = pd.DataFrame({
    "Area_Sqft": [2200],
    "Bedrooms": [3],
    "Bathrooms": [2],
    "House_Age": [10],
    "Distance_To_City": [8.5],
    "Location_Score": [8]
})

predicted_price = model.predict(new_house)

print("House Details:")
print(new_house)

print(f"\nPredicted House Price: ${predicted_price[0]:,.2f}")