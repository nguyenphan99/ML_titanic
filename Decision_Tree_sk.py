from Pre_Processing_Data import *
from sklearn import tree
from sklearn.metrics import accuracy_score
import numpy as np



# Đọc dữ liệu
X_train, y_train, X_test, y_test = Load_Data('F:/UIT/ML/Titanic/My_Data.txt')

clf = tree.DecisionTreeClassifier()
clf = clf.fit(X_train, y_train)

pre = clf.predict(X_test)
accuracy = accuracy_score(y_test, pre)

print(accuracy)