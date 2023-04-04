#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Author: Shen Chan Huang
#dataset: MNIST_HW4
#Task: Apply SVM with 3 kernels for classification problem
#Validation: Use 5-fold cross-validation


# In[37]:


import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import KFold
from sklearn.metrics import classification_report, accuracy_score

file_name = 'MNIST_HW4.csv'
df = pd.read_csv(file_name)


# In[20]:


#set response variable
y = df['label']
X = df.drop('label', axis=1)


# In[59]:


#define standard scaler
std_scaler = StandardScaler()

#define minmax scaler
minmax_scaler = MinMaxScaler()


# In[93]:


#define models
model = SVC(kernel='linear')
#model = SVC(kernel='poly')
#model = SVC(kernel='rbf')


# In[57]:


#define K-Fold Cross-Validation
kf = KFold(n_splits=5, shuffle=True, random_state=42)


# In[94]:


#loop over splits
accuracy = list()
for fold, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, y_train = X.iloc[train_index], y.iloc[train_index]
    X_test, y_test = X.iloc[test_index], y.iloc[test_index]
#    X_train_scaled = pd.DataFrame(minmax_scaler.fit_transform(X_train))
#    X_test_scaled = pd.DataFrame(minmax_scaler.transform(X_test))
#    model.fit(X_train_scaled, y_train)
#    predictions = model.predict(X_test_scaled)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy.append(accuracy_score(y_test, predictions))
#    report = pd.DataFrame(classification_report(y_test, predictions, output_dict=True, digits=4))
print(accuracy)
print(np.mean(accuracy))


# In[ ]:




