import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.cluster import KMeans

# Convert file csv to data frame
df = pd.read_csv('Mall_Customers.csv')

# Show 3 row first
a = df.head(3)
print(a)
print('\n')

# Change name Column
df = df.rename(columns={'Gender': 'gender', 'Age': 'age',
                        'Annual Income (k$)': 'annual_income',
                        'Spending Score (1-100)': 'spending_score'})

# Change data category to data numeric
df['gender'].replace(['Female', 'Male'], [0, 1], inplace=True)

# show data is preprocessed
b = df.head(3)
print(b)

# remove column customer id dan gender
X = df.drop(['CustomerID', 'gender'], axis=1)

# make list yang berisi inertia
clusters = []
for i in range(1, 11):
    km = KMeans(n_clusters=i).fit(X)
    clusters.append(km.inertia_)

# make plot inertia
fig, ax = plt.subplots(figsize=(8, 4))
sns.lineplot(x=list(range(1, 11)), y=clusters, ax=ax)
ax.set_title('Cari Elbow')
ax.set_xlabel('Clusters')
ax.set_ylabel('Inertia')

# make object KMeans
km5 = KMeans(n_clusters=5).fit(X)

# add column label in data set
X['Labels'] = km5.labels_

# make plot KMeans with 5 cluster
plt.figure(figsize=(8, 4))
sns.scatterplot(X['annual_income'], X['spending_score'], hue=X['Labels'],
                palette=sns.color_palette('hls', 5))
plt.title('KMeans with 5 Cluster')
plt.show()
