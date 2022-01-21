
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('weather.csv')
X = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Decision Tree Classification
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
classifier.fit(X_train, y_train)

# Naive Bayes Classification
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# calculating accuracy
from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)

# calculating recall
from sklearn.metrics import recall_score
recall=recall_score(y_test,y_pred,average="binary")

# calculating precision
from sklearn.metrics import precision_score
precision=precision_score(y_test,y_pred,average="binary")

# calculating f1 score 
from sklearn.metrics import f1_score
f1score=f1_score(y_test,y_pred,average="binary")

# print('For Decision Tree: \n')
# print('For Random Forest: \n')
print('For Naive Bayes: \n')

print('Accuracy is : %.2f'%accuracy)
print('Recall is : %.2f'%recall)
print('precision is :%.2f'%precision)
print('f1 score : %.2f'%f1score)

