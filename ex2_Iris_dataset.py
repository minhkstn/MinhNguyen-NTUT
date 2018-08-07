import numpy as np
from sklearn import neighbors, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

iris = datasets.load_iris()
iris_X = iris.data
iris_y = iris.target

# split data
X_train, X_test, y_train, y_test = train_test_split(iris_X, iris_y, test_size=50)
print('Labels: ', np.unique(y_train))
print('Train size:', X_train.shape[0])

# build model
model = neighbors.KNeighborsClassifier(n_neighbors=10,p=2)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

print("Acc: {}".format(accuracy_score(y_test, y_pred)) )