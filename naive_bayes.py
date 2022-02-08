import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

dataset = pd.read_csv("SocialNetworkAds.csv")
#print(dataset.head())
X = dataset.iloc[:, [2,3]]
y = dataset.iloc[:,4]

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y, test_size = 0.25,random_state = 0)

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train=sc.fit_transform(X_train)
X_test=sc.fit_transform(X_test)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

y_predict = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_predict,y_test))

from sklearn import metrics
print(metrics.classification_report(y_predict,y_test))

