import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# House area (in 100 sq.ft) vs price (in lakhs)
area = np.array([4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15]).reshape(-1, 1)
price = np.array([22, 27, 29, 36, 40, 43, 51, 53, 60, 63, 70, 72])

X_train, X_test, y_train, y_test = train_test_split(
    area, price, test_size=0.25, random_state=0)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("Slope (m)     :", round(model.coef_[0], 4))
print("Intercept (c) :", round(model.intercept_, 4))
print("MSE           :", round(mean_squared_error(y_test, y_pred), 4))
print("R2 score      :", round(r2_score(y_test, y_pred), 4))
print("Predicted price for 16 (1600 sq.ft):",
      round(model.predict([[16]])[0], 2), "lakhs")

plt.scatter(area, price, color="green", label="Actual price")
plt.plot(area, model.predict(area), color="orange", label="Regression line")
plt.xlabel("Area (100 sq.ft)")
plt.ylabel("Price (lakhs)")
plt.title("Simple Linear Regression: Area vs Price")
plt.legend()
plt.show()
