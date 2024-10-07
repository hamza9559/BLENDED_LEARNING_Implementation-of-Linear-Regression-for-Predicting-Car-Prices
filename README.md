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
```
/*
 Program to implement linear regression model for predicting car prices and test assumptions.
Developed by: HAMZA FAROOQUE 
RegisterNumber:  212223040054
*/
# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = {
    'Age': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Mileage': [5000, 15000, 25000, 35000, 45000, 55000, 65000, 75000, 85000, 95000],
    'Price': [20000, 18500, 17500, 16500, 15500, 14500, 13500, 12500, 11500, 10500]
}

df = pd.DataFrame(data)


X = df[['Age', 'Mileage']]

y = df['Price']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)

print(f"Predicted Prices: {y_pred}")
print(f"Mean Squared Error: {mse}")

plt.scatter(y_test, y_pred)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color='red', linewidth=2)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Actual vs Predicted Car Prices')
plt.show()

```

## Output:
```
Predicted Prices: [16000. 11000.]

Mean Squared Error: 1000000.0


```

![image](https://github.com/user-attachments/assets/3f81cb76-d009-49d0-b836-e0734423ffac)



## Result:
Thus, the program to implement a linear regression model for predicting car prices is written and verified using Python programming, along with the testing of key assumptions for linear regression.
