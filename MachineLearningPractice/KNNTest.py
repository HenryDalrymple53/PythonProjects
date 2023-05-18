import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from KNN import KNN

cmap = ListedColormap(['#FF0000','#00FF00', '#0000FF'])

iris = datasets.load_iris()
x,y = iris.data, iris.target
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.2, random_state=9876)

plt.figure()




classifier = KNN(k=3)
classifier.fit(x_train,y_train)
predictions = classifier.predict(x_test)

plt.scatter(x_test[:,0], x_test[:,1],c=(y_test==predictions), cmap=cmap, edgecolor = 'k')

plt.show()
accuracy = np.sum(predictions == y_test) / len(y_test)
print(accuracy)


 