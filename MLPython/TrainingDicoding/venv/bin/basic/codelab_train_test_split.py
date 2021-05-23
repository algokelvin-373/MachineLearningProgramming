import sklearn
from sklearn import datasets
from sklearn.model_selection import train_test_split

# Reference
# https://stackoverflow.com/questions/32675024/getting-pycharm-to-import-sklearn

# Step 1 - load iris data set
iris = datasets.load_iris()

# Step 2 - Create Attribute iris - x: data , y: target
x = iris.data
y = iris.target

# Step 3 - Create Data Set Training dan Testing
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# Step 4 - Print Data Training
print("----Training Data----")
print("x : ")
print(x_train)
print("y : ")
print(y_train)

# Step 5 - Print Data Testing
print("\n----Testing Data----")
print("x : ")
print(x_test)
print("y : ")
print(y_test)
