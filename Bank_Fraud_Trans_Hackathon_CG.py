# -*- coding: utf-8 -*-
"""
Created on Sat Sep  8 16:13:39 2018

@author: sparachi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pt


import os
import sys
print (os.getcwd())
os.chdir('D:\Koushik\CG\Hackathon\paysim1')
print (os.getcwd())

Data = pd.read_csv('PaySim_Log1.csv')

Data_X = pd.DataFrame(Data.iloc[:, 1:9])
Data_y = pd.DataFrame(Data.iloc[:, 9:])

Data_X.head()
#Data.describe()

Vars=Data.isnull().sum()


#Data=Data.reshape(-1,1)

# Feature Scaling
Data_SC = pd.DataFrame(Data.iloc[:, 1:9])
Data_SC.head()
Data_SC.drop(Data_SC.columns[[0,2,5]], axis=1, inplace=True)
Data_SC.head()
SC_Columns =['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
from sklearn.preprocessing import scale as sc
Data_SC =pd.DataFrame(sc.fit_transform(Data_SC),columns= SC_Columns)



'''sc = StandardScaler()
Data_SC =pd.DataFrame(sc.fit_transform(Data_SC),columns= SC_Columns)
Data_SC =pd.DataFrame(sc.transform(Data_SC),columns= SC_Columns)
#type(Data_SC)
Data_X.head()
Data_SC.head()
sc.fit_transform()
##################################################################
Data_Merge1= pd.merge(left=Data_X.head(),Data_SC.head(),how='outer', on='amount',copy=True)

Data_SCed = pd.concat([Data_SC,Data_X],axis=0)
Data_SCed.head()
Data_X1 = pd.DataFrame(Data_SC)

Data_X1= Data_SC.copy(deep=True)
Data_X1.head()'''
'''
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
for col in Data_X:
#        if str(Data_X[col].dtype) == 'int64':
#           sc.fit_transform([Data_X[col]]) 
#           Data_X = sc.transform([Data_X[col]])
        if (str(Data_X[col].dtype) == 'float64'):
           sc.fit_transform([Data_X[col]]) 
           Data_X = sc.transform([Data_X[col]])
            
        else:
           pass
#sc.fit(Data) 
#df_X = sc.transform(df_X)'''

#plt.scatter(Data[:, 0], )
#plt.scatter(Data[:, :], Data[:, :])
Data.dtypes
Data_X.dtypes

Data_X['nameOrig'].dtype
#Data.plot(kind='scatter', x='step', y='amount', )#s=df.col3)

#Data.plot(kind='scatter', x='oldbalanceOrg', y='amount', )

Data.plot()

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(Data_SC, Data_y, test_size = 0.25, random_state = 123)

# Fitting SVM to the Training set
from sklearn.svm import SVC
classifier = SVC(kernel = 'rbf', random_state = 0)
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)


# Visualising the Training set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Training set)')
plt.xlabel('Fraud Tran')
plt.ylabel('Estimated Fraudness')
plt.legend()
plt.show()

# Visualising the Test set results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.75, cmap = ListedColormap(('red', 'green')))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                c = ListedColormap(('red', 'green'))(i), label = j)
plt.title('SVM (Test set)')
plt.xlabel('Fraud Tran')
plt.ylabel('Estimated Fraudness')
plt.legend()
plt.show()