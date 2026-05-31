# House Price Prediction

## Overview
This project uses machine learning to predict real estate house prices based on property features.

The dataset includes house area, number of bedrooms, bathrooms, house age, distance to the city center, location score, and price.

## Features
- Generates a real estate dataset
- Trains a machine learning regression model
- Predicts house prices using property information
- Evaluates model performance using MAE and R2 Score
- Saves the trained model
- Creates data visualization charts

## Technologies Used
- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Joblib

## Project Structure
```text
house-price-prediction/
├── main.py
├── predict.py
├── requirements.txt
├── README.md
├── data/
│   └── house_prices.csv
├── charts/
│   ├── area_vs_price.png
│   ├── actual_vs_predicted.png
│   └── price_by_location_score.png
└── models/
    └── house_price_model.pkl
```
## How to run
1. Install requirements:
pip install -r requirements.txt 

2. Train the model:
python main.py

3. Predict a house price
python predict.py

## Model Evaluation
The model is evaluated using:
- Mean Ansolute Error
- R2 Score

## What I learned
This project helped me pratice machine learning regression, model evaluation, data visualization and saving trained models.

## Author
Created by Noor Hussein