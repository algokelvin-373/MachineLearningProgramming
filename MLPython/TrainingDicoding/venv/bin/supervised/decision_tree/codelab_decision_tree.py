import pandas as pd
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
from sklearn.tree import export_graphviz

# Step 1 - Read file iris.csv
# https://www.datacamp.com/community/blog/python-pandas-cheat-sheet?utm_source=adwords_ppc&utm_campaignid=12492439802&utm_adgroupid=122563403481&utm_device=c&utm_keyword=panda%20package%20python&utm_matchtype=b&utm_network=g&utm_adpostion=&utm_creative=504158804614&utm_targetid=aud-392016246653:kwd-614516587896&utm_loc_interest_ms=&utm_loc_physical_ms=9072592&gclid=Cj0KCQjw16KFBhCgARIsALB0g8KVWMbu2mVEPKV5uOJAAG1BlJ_77eQjKYB0DM0mAU65snWvmXM4D20aAoU5EALw_wcB
iris = pd.read_csv('decision_tree/Iris.csv')

# Step 2 - Get Information 5 Data Set Iris
iris_data_5 = iris.head()
print(iris_data_5)

# Step 3 - Remove column which isn't important
iris_drop = iris.drop('Id', axis=1, inplace=True)

# Step 4 - separate attribute dan label
X = iris[['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm']]
y = iris['Species']

# Step 5 - Create Model Decision Tree
tree_model = DecisionTreeClassifier()

# Step 6 - Do Training model with data
tree_model.fit(X, y)

# Step 7 - Prediction model with tree_model.predict([[SepalLength, SepalWidth, PetalLength, PetalWidth]])
test1 = tree_model.predict([[6.2, 3.4, 5.4, 2.3]])
test2 = tree_model.predict([[4.7, 3.0, 1.3, 0.5]])
print(test1)
print(test2)

# Step 8 - Create Visualization Decision Tree
export_graphviz(
    tree_model,
    out_file="decision_tree/iris_tree.dot",
    feature_names=['SepalLengthCm', 'SepalWidthCm', 'PetalLengthCm', 'PetalWidthCm'],
    class_names=['Iris-setosa', 'Iris-versicolor', 'Iris-virginica'],
    rounded=True,
    filled=True
)
