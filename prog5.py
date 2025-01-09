from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

iris = load_iris()
X = iris.data
y = iris.target

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=50)

scalar = StandardScaler()
X_train = scalar.fit_transform(X_train)
X_test = scalar.transform(X_test)

# Train KNN classifier
k = 5
knn = KNeighborsClassifier(n_neighbors=k)
knn.fit(X_train,y_train)

y_pred = knn.predict(X_test)

accuracy = accuracy_score(y_test, y_pred) * 100

print(f'Accuracy of K-Nearest Neighbors classifier on Iris dataset: {accuracy:.2f}%')

import pandas as pd
iris = load_iris()
iris.feature_names
iris.target_names
df = pd.DataFrame(iris.data, columns=iris.feature_names)

df['target'] = iris.target
df[df.target==1].head()
df0 = df[df.target==0]
df1 = df[df.target==1]
df2 = df[df.target==2]
print(df2.head())

import matplotlib.pyplot as plt
plt.xlabel('Sepal Length')
plt.ylabel('Sepal Width')
plt.scatter(df0['sepal length (cm)'], df0['sepal width
(cm)'],color="green",marker='+')
plt.scatter(df1['sepal length (cm)'], df1['sepal width
(cm)'],color="blue",marker='.')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')
plt.scatter(df0['petal length (cm)'], df0['petal width
(cm)'],color="green",marker='+')
plt.scatter(df1['petal length (cm)'], df1['petal width
(cm)'],color="blue",marker='.')
plt.scatter(df2['petal length (cm)'], df2['petal width
(cm)'],color="red",marker='.')
from sklearn.model_selection import train_test_split
X = df.drop(['target'], axis='columns')
y = df.target
X_train, X_test, y_train, y_test =
train_test_split(X,y,test_size=0.2)
len(X_train)
len(X_test)
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
knn.score(X_test, y_test)