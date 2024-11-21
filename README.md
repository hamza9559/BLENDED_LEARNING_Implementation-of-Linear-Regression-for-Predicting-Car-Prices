# Implementation-of-Linear-Regression-for-Predicting-Car-Prices
## AIM:
To write a program to predict car prices using a linear regression model and test the assumptions for linear regression.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1:Data Preparation: Load the dataset containing features (car age, mileage) and the target (car price), then split it into training and testing sets.

2:Model Training: Create a Linear Regression model and train it using the training data (age, mileage) to predict car prices.

3:Prediction: Use the trained model to predict car prices on the test data and calculate the mean squared error to evaluate the accuracy.

4:Visualization: Plot a graph comparing actual car prices with the predicted prices for visual validation.

## Program:
```c

/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: HAMZA FAROOQUE 
RegisterNumber: 212223040054
*/
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.api as sm

# Load the dataset
file_path = 'CarPrice.csv'
df = pd.read_csv(file_path)

# Select relevant features and target variable
X = df[['enginesize', 'horsepower', 'citympg', 'highwaympg']]  # Features
y = df['price']  # Target variable

# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Model Evaluation
print("Coefficients:", model.coef_)
print("Intercept:", model.intercept_)
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R-squared:", r2_score(y_test, y_pred))

# 1. Assumption: Linearity
plt.scatter(y_test, y_pred)
plt.title("Linearity: Observed vs Predicted Prices")
plt.xlabel("Observed Prices")
plt.ylabel("Predicted Prices")
plt.show()

# 2. Assumption: Independence (Durbin-Watson test)
residuals = y_test - y_pred
dw_test = sm.stats.durbin_watson(residuals)
print(f"Durbin-Watson Statistic: {dw_test}")

# 3. Assumption: Homoscedasticity
sns.scatterplot(x=y_pred, y=residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.title("Homoscedasticity: Residuals vs Predicted Prices")
plt.xlabel("Predicted Prices")
plt.ylabel("Residuals")
plt.show()

# 4. Assumption: Normality of residuals
sns.histplot(residuals, kde=True)
plt.title("Normality: Histogram of Residuals")
plt.show()

sm.qqplot(residuals, line='45')
plt.title("Normality: Q-Q Plot of Residuals")
plt.show()

# Insights
print("Check these outputs to verify assumptions for linear regression.")

```

## Output:

![image](https://github.com/user-attachments/assets/8de47e4b-1661-4b50-836c-7e004eac29ec)

![image](https://github.com/user-attachments/assets/2ed95d45-694c-4fe6-99df-8f2957322089)

![image](https://github.com/user-attachments/assets/4a9c242f-e725-4abd-b510-b8ac536db4fd)

![image](https://github.com/user-attachments/assets/27f8ab25-923d-445d-a62e-b2003e4e7691)

![image](https://github.com/user-attachments/assets/b72e5c81-7874-47f1-800d-b0aede591ca0)



## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
