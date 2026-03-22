import pandas as pd
import numpy as np
from statsmodels.formula.api import ols
from sklearn.metrics import mean_squared_error
import pickle

# Load dataset
df = pd.read_csv("cardekho_dataset.csv")

print("Dataset Shape:", df.shape)
print(df.head())

def remove_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    lower = Q1 - 1.5 * IQR
    upper = Q3 + 1.5 * IQR

    return df[(df[column] >= lower) & (df[column] <= upper)]

columns = ['selling_price', 'km_driven', 'mileage', 'engine', 'max_power']

df_clean = df.copy()

for col in columns:
    df_clean = remove_outliers(df_clean, col)

print("Before Cleaning:", df.shape)
print("After Cleaning:", df_clean.shape)

# Train OLS model
model = ols(
    'selling_price ~ vehicle_age + km_driven + brand + seller_type + fuel_type + transmission_type + mileage + engine + max_power + seats',
    data=df_clean
).fit()

print(model.summary())

# Make predictions
predictions = model.predict(df_clean)

r2 = model.rsquared
rmse = np.sqrt(mean_squared_error(df_clean['selling_price'], predictions))

print("\n=== Model Performance ===")
print("R2:", r2)
print("RMSE:", rmse)

# Save model
pickle.dump(model, open("model.pkl", "wb"))
print("\nModel saved successfully as model.pkl!")

print("\nUnique Brands:", df['brand'].unique())
