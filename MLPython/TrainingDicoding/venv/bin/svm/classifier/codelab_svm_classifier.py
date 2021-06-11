import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC

df = pd.read_csv('diabetes.csv')

head = df.head()
info = df.info()
print(head)
print(info)

# separate the attributes in the data set and store them in a variable
X = df[df.columns[:8]]

# separate the labels on the data set and store them in a variable
y = df['Outcome']

# standardize the values ​​of the data set
scaler = StandardScaler()
scaler.fit(X)
X = scaler.transform(X)

# separate data for training and testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)

# create object SVC dan call function fit to training model
clf = SVC()
clf.fit(X_train, y_train)

# Show score accuracy prediction
accuracy = clf.score(X_test, y_test)
print(accuracy)
