# -*- coding: utf-8 -*-
"""
Created on Tue Sep 18 23:10:18 2018

@author: sparachi
"""
###################################################################################
###Importing Essential base Libraries::
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import sys
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix,classification_report

##Setting workning directory::
print (os.getcwd())
os.chdir('D:\Koushik\CG\Hackathon\HealthCare_Hackathone_DataSet_CG\Capstone Project')
print (os.getcwd())

###loading Data sets:::
Train_Data= pd.read_csv('train Data.csv')
Train_labels= pd.read_csv('train labels.csv')

############################
## Preprocessing of Data includes handling missing values by eliminating those features who's missing count is >7000,
##Converting Categorical features using dummies,Scalling Numerical features using Standered scalling,

#Verifying for missing values:::
Train_Data.dtypes
Train_Data.isnull().sum()
Train_labels.isnull().sum()

##Dropping columns who's sum of missing values count is eqauls to 7000.::
Train_Data.dropna(how='all',inplace=True,axis=1,thresh=7000)

##Imputing data with mean median mode respectively::
for col in Train_Data:
    if Train_Data[col].isnull().sum() > 0:
        if str(Train_Data[col].dtype) == 'category':
           Train_Data[col] = Train_Data[col].fillna(value = Train_Data[col].mode()[0])
        elif (str(Train_Data[col].dtype) == 'int64'):
           Train_Data[col] = Train_Data[col].fillna(value = Train_Data[col].mean())
        elif (str(Train_Data[col].dtype) == 'float64'):
           Train_Data[col] = Train_Data[col].fillna(value = Train_Data[col].mean())
        elif (str(Train_Data[col].dtype) == 'object'):
            Train_Data[col] = Train_Data[col].fillna(value = Train_Data[col].mode()[0])
    else:
        pass
            
           
Train_Data.head()
type(Train_Data)



##Handling Categorical features using dummies function::
Train_Data.head()
Train_Data=pd.get_dummies(Train_Data,drop_first=True)
Train_Data.head()

############################################################
#Feature Elimination::
###Building Random Forest Classifier to find feature_importances:

from sklearn.ensemble import RandomForestClassifier

y_Train = pd.DataFrame(Train_labels.iloc[:,1])
X_Train = pd.DataFrame(Train_Data.iloc[:, 0:])


X_train, X_test, y_train, y_test = train_test_split(X_Train, y_Train,
                                                    test_size=0.20)

rf = RandomForestClassifier(n_estimators=100,criterion='gini',max_depth=50 ,max_features='auto',n_jobs=-1) 
rf.fit(X_train, y_train)
rf.score(X_test, y_test)    
feature_importances = pd.DataFrame(rf.feature_importances_,
                                   index = X_train.columns,
                                    columns=['importance']).sort_values('importance',ascending=False)


print(feature_importances.head(n=20))

##Collected top 50 Unique features list of important/major contributors of further model building for all 14 target variables.:::  
Top_50_feature = ['c_1259_n', 'n_0108', 'n_0078', 'n_0002', 'c_1259_i', 'c_1029_s', 'c_1033_h', 'c_1190_s',
                  'c_0835_v',  'c_1225_n', 'id', 'n_0012', 'c_1247_o', 'c_1172_s', 'o_0176', 'c_1109_x',
                  'c_0391_g', 'c_0944_i', 'n_0086', 'n_0102','c_1003_b','c_1190_s','c_1247_o','n_0012',
                  'c_0944_i','o_0176','c_1172_s','c_1109_x','c_0965_h','c_1045_h','n_0102','o_0141',
                  'n_0083','o_0201','c_1259_e','c_0391_c','o_0279','c_1225_j','n_0109','c_1029_e','c_1172_w',
                  'o_0301','o_0125','c_0835_p','c_0965_i','c_1033_o','o_0175','n_0100','c_0554_b','n_0064',
                  'o_0168','o_0268','o_0270','c_1259_b','o_0120','o_0223','c_1259_c','c_1316_b','o_0241',
                  'c_0591_g','c_0749_m']  



###########################################For Train_Data.csv##################
##Spilting Data for Trainning and Test Data on Train_Data.csv and sampling using Statifies samopling.
######Model For First Target Variable.
y_Train = pd.DataFrame(Train_labels.iloc[:,1])
X_Train = pd.DataFrame(Train_Data.iloc[:, :],columns= Top_50_feature)

#Here considering y_Train[:,1] only, because we are about to build base logistic model
X_train, X_test, y_train, y_test = train_test_split(X_Train, y_Train, 
                                                    test_size=0.20,
                                                    random_state=143)

### Building Logisticregression model::
model = LogisticRegression(C=1.0,class_weight='balanced', solver='newton-cg', multi_class='ovr' )
model = model.fit(X_train, y_train)
y_pred = model.predict(X_test)

## Classification report::
print(classification_report(y_test, y_pred))

####Implementing OneVsRestClassifier to habndel all 14 target variables::::
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import jaccard_similarity_score

y_Train = pd.DataFrame(Train_labels.iloc[:,1:])
X_Train = pd.DataFrame(Train_Data.iloc[:, :],columns= Top_50_feature)


X_train, X_test, y_train, y_test = train_test_split(X_Train, y_Train,
                                                    test_size=0.20,
                                                    random_state=143)

ovr = OneVsRestClassifier(LogisticRegression(C=1.0,class_weight='balanced', solver='newton-cg', multi_class='ovr'))
ovr.fit(X_train, y_train)
y_pred_ovr = ovr.predict(X_test)
ovr_jaccard_score = jaccard_similarity_score(y_test, y_pred_ovr)
print (ovr_jaccard_score)
## Classification report::
print(classification_report(y_test, y_pred_ovr))


# classifier chain model::
# Fit an ensemble of logistic regression classifier chains and take the
# take the average prediction of all the chains.

from sklearn.multioutput import ClassifierChain
from sklearn.metrics import jaccard_similarity_score

chains = [ClassifierChain(LogisticRegression(), order='random', random_state=i)
          for i in range(10)]
for chain in chains:
    chain.fit(X_train, y_train)

y_pred_chains = np.array([chain.predict(X_test) for chain in
                          chains])
chain_jaccard_scores = [jaccard_similarity_score(y_test, y_pred_chain >= .5)
                        for y_pred_chain in y_pred_chains]

y_pred_ensemble = y_pred_chains.mean(axis=0)
ensemble_jaccard_score = jaccard_similarity_score(y_test,
                                                  y_pred_ensemble >= .5)


############ Visualization for demo   #####################

model_scores = [ovr_jaccard_score] + chain_jaccard_scores
model_scores.append(ensemble_jaccard_score)

model_names = ('Independent',
               'Chain 1',
               'Chain 2',
               'Chain 3',
               'Chain 4',
               'Chain 5',
               'Chain 6',
               'Chain 7',
               'Chain 8',
               'Chain 9',
               'Chain 10',
               'Ensemble')

x_pos = np.arange(len(model_names))

# Plot the Jaccard similarity scores for the independent model, each of the
# chains, and the ensemble (note that the vertical axis on this plot does
# not begin at 0).

fig, ax = plt.subplots(figsize=(7, 4))
ax.grid(True)
ax.set_title('Classifier Chain Ensemble Performance Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation='vertical')
ax.set_ylabel('Jaccard Similarity Score')
ax.set_ylim([min(model_scores) * .9, max(model_scores) * 1.1])
colors = ['r'] + ['b'] * len(chain_jaccard_scores) + ['g']
ax.bar(x_pos, model_scores, alpha=0.5, color=colors)
plt.tight_layout()
plt.show()



