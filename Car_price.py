# Car_price.py
# Train model and save car_model.pkl

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import pickle
from datetime import datetime

# Load dataset
df = pd.read_csv("car_price_dataset.csv")
print(df.head())

# Data Preprocessing
current_year = datetime.now().year
df['Car_Age'] = current_year - df['Year']
df.drop(['Year'], axis=1, inplace=True)
df = pd.get_dummies(df, drop_first=True)

# Save column order for correct prediction in Flask
feature_columns = df.drop("Selling_Price", axis=1).columns.tolist()
pickle.dump(feature_columns, open("feature_columns.pkl", "wb"))

X = df.drop("Selling_Price", axis=1)
y = df["Selling_Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
pred = model.predict(X_test)
print("R2 Score:", r2_score(y_test, pred))

# Save model
pickle.dump(model, open("car_model.pkl", "wb"))
print("✔ Model saved as car_model.pkl")
