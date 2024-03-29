import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('Social_Network_Ads.csv')

df = data
df.head()
print(df.head())

df.info()

data = df.drop(columns=['User ID'])

data = pd.get_dummies(data)
print(data)

predictions = ['Age' , 'EstimatedSalary' , 'Gender_Female' , 'Gender_Male']
X = data[predictions]
y = data['Purchased']

scaler = StandardScaler()
scaler.fit(X)
scaled_data = scaler.transform(X)
scaled_data = pd.DataFrame(scaled_data, columns= X.columns)
scaled_data.head()

X_train, X_test, y_train, y_test = train_test_split(scaled_data, y, test_size=0.2, random_state=1)

model = linear_model.LogisticRegression()
model.fit(X_train, y_train)

model.score(X_test, y_test)
