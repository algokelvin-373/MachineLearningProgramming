import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVR

df = pd.read_csv('Salary_Data.csv')

head = df.head()
info = df.info()
print(head)
print(info)

# separate attributes and labels
X = df['YearsExperience']
y = df['Salary']

# change attribute
X = X[:, np.newaxis]

# build model with parameter C, gamma, dan kernel
model = SVR(C=1000, gamma=0.05, kernel='rbf')

# training model with function fit
model.fit(X, y)

# visualization model
plt.scatter(X, y)
plt.plot(X, model.predict(X))
