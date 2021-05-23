import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression

# Data Sample
bedrooms = np.array([1, 1, 2, 2, 3, 4, 4, 5, 5, 5])  # data n Room
house_price = np.array([15000, 18000, 27000, 34000, 50000, 68000, 65000, 81000, 85000, 90000])  # data price Room

# Show scatter plot from data set
plt.scatter(bedrooms, house_price)
plt.show()

# Training model with Linear Regression.fit()
bedrooms = bedrooms.reshape(-1, 1)
lg = LinearRegression()
lg.fit(bedrooms, house_price)

# Show plot correlation between total room and price rumah
plt.scatter(bedrooms, house_price)
plt.plot(bedrooms, lg.predict(bedrooms))
plt.show()
