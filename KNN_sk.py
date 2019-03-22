from Pre_Processing_Data import *
import numpy as np 
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Đọc dữ liệu
X_train, y_train, X_test, y_test = Load_Data('F:/UIT/ML/Titanic/My_Data.txt')

neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(X_train, y_train)

pre = neigh.predict(X_test)
accuracy = accuracy_score(y_test, pre)

print(accuracy)