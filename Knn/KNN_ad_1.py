print('社群網路廣告,0代表不買,1代表購買')
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
dataset=pd.read_csv('Social_Network_Ads.csv')
X=dataset.iloc[:,[2,3]].values
y=dataset.iloc[:,4].values
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(
X,y,test_size=0.25,random_state=0)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
print(X_train)
X_train=sc.fit_transform(X_train)
print(X_train)
print(X_test)
X_test=sc.transform(X_test)
print(X_test)
from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=2)
classifier.fit(X_train,y_train)