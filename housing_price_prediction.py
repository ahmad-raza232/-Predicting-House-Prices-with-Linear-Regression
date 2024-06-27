# Housing Price Prediction

This project predicts housing prices using linear regression.

## Dataset

The dataset used is the `housing.csv` dataset from Seaborn.

## Requirements

- numpy
- pandas
- matplotlib
- scikit-learn

## Usage

Run the `housing_price_prediction.py` script to train the model and see the predictions.

## Code

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Load dataset
url = "https://raw.githubusercontent.com/mwaskom/seaborn-data/master/housing.csv"
df = pd.read_csv(url)

# Data preprocessing
df.dropna(inplace=True)

# Defining features and target variable
X = df[['GrLivArea', 'YearBuilt', 'TotalBsmtSF']]
y = df['SalePrice']

# Splitting data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Training the model
model = LinearRegression()
model.fit(X_train, y_train)

# Making predictions
y_pred = model.predict(X_test)

# Evaluating the model
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Plotting the results
plt.scatter(y_test, y_pred)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Prices')
plt.show()
