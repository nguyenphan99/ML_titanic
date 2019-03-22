from Pre_Processing_Data import *
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score


# Đọc dữ liệu
X_train, y_train, X_test, y_test = Load_Data('F:/UIT/ML/Titanic/My_Data.txt')

gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)

pre = gnb.predict(X_test)
accuracy = accuracy_score(y_test, pre)

print(accuracy)