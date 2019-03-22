from Pre_Processing_Data import *
import numpy as np 

# Đọc dữ liệu
X_train, y_train, X_test, y_test = Load_Data('F:/UIT/ML/Titanic/My_Data.txt')


def distance(point_1, point_2):
	distance = 0
	length = point_1.shape[0]
	for i in range(length):
		distance += (point_1[i] - point_2[i])**2
	return distance

print(X_train[0])
print(X_train[1])

print(distance(X_train[0], X_train[1]))