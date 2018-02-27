# Classification of MNIST data using various classifiers and comparing the results

import pandas as pd
import numpy as np

from sklearn.datasets import fetch_mldata
mnist = fetch_mldata("MNIST original")
print(mnist.data.shape)
print(mnist.target.shape)

from sklearn.model_selection import train_test_split

train_im, test_im, train_lab, test_lab = train_test_split(
    mnist.data, mnist.target, test_size=1/7.0, random_state=0)
for index, (image, label) in enumerate (zip(train_im[0:5], train_lab[0:5])):
    plt.subplot(1,5,index+1)   
    plt.imshow(np.reshape(image,(28,28)), cmap=plt.cm.gray)     
    plt.title('Training: %i\n'%label, fontsize=20)
    plt.show()

# Logistic Regression with all pixels as features
from sklearn.linear_model import LogisticRegression
logisticreg = LogisticRegression(solver = 'lbfgs')
logisticreg.fit(train_im,train_lab)
predictions=logisticreg.predict(test_im)
#accuracy
score=logisticreg.score(test_im,test_lab)
print(score)
from sklearn import metrics
cm=metrics.confusion_matrix(test_lab,predictions)
print(cm)

#knn classifier with k=5

from sklearn.neighbors import KNeighborsClassifier
knn=KneighborsClassifier(n_neighbors=5)
knn.fit(train_im,train_lab)
predictions_knn=knn.predict(test_im)
metrics.accuracy_score(test_lab,predictions_knn)
cm_knn=metrics.confusion_matrix(test_lab,predictions)
print(cm_knn)

#Accuracy with cross validation
knn = KNeighborsClassifier(n_neighbors=5)
scores = cross_val_score(knn, mnist.data, mnist.target, cv=10, scoring='accuracy')
print(scores)
print(scores.mean())

#Find optimal Value of k
k_range = range(1, 31)
k_scores = []
for k in k_range:
    knn = KNeighborsClassifier(n_neighbors=k)
    scores = cross_val_score(knn, train_im, train_lab, cv=10, scoring='accuracy')
    k_scores.append(scores.mean())
print(k_scores)
print(max(k_scores))
kfinal=(np.argmax(k_scores))+1
knn = KNeighborsClassifier(n_neighbors=kfinal)
knn.fit(train_im,train_lab)
predictions_knn=knn.predict(test_im)
metrics.accuracy_score(test_lab,predictions_knn)
cm_knn=metrics.confusion_matrix(test_lab,predictions)
print(cm_knn)

# SVM
from sklearn import svm
clf = svm.SVC(gamma=0.001, C=100.)
clf.fit(train_im, train_lab)
clf.predict(digits.data[-1:])

# Decision trees
from sklearn import tree
decision = tree.DecisionTreeClassifier()
clf_decision = decision.fit(train_im,train_lab)
clf_decision.predict(test_im)



    




