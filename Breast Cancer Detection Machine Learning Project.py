#!/usr/bin/env python
# coding: utf-8

# # Breast Cancer Detection

# In[37]:


import warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#load the data
df=pd.read_csv("data.csv")
df


# In[3]:


#count the number of rows and columns in the data set
df.shape


# In[4]:


df.info()


# In[5]:


#count the number of empty (NaN)Values in each column
df.isna().sum()


# In[6]:


#Drop the column with the all mising values
df=df.dropna(axis=1)


# In[7]:


#Get the new count of the number of rows and columns
df.shape


# In[8]:


#Get a count of the number of Maligeant(M)or Benign(B)cells 
df["diagnosis"].value_counts()


# # Visualize the data

# In[9]:


#visualise the count
sns.countplot(df["diagnosis"],label="count")


# In[10]:


#look at the data types to see which col need to be encoded
df.dtypes


# In[11]:


#encode the categorical data values
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df.iloc[:,1]=le.fit_transform(df.iloc[:,1].values)


# In[12]:


#create a pair plot
sns.pairplot(df.iloc[:,1:5],hue="diagnosis")


# In[13]:


plt.figure(figsize=(15,15))
sns.distplot(df["radius_mean"])


# In[14]:


#print the first 5 rows of the new data
df.head(5)


# In[15]:


df.corr()


# In[16]:


#Visualize the correlation
plt.figure(figsize=(25,25))
sns.heatmap(df.corr(),annot=True)


# In[17]:


#split the data into independant(x) and dependant(y) data set
x=df.iloc[:,2:31]
y=df.iloc[:,1]


# In[18]:


x


# In[19]:


y


# In[20]:


#split the data set into training and testing 
from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.25,random_state=0)


# In[21]:


xtrain


# In[22]:


#Scale the data (feature scaling)
from sklearn.preprocessing import StandardScaler
sc=StandardScaler()
xtrain=sc.fit_transform(xtrain)
xtest=sc.fit_transform(xtest)


# In[23]:


xtrain


# In[24]:


from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier



# In[25]:


from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import classification_report


# In[26]:


def mymodel(model):
    model.fit(xtrain,ytrain)
    ypred=model.predict(xtest)
    
    train=model.score(xtrain,ytrain)
    test=model.score(xtest,ytest)
    
    print(f"Training accuracy:{train}\nTesting accuracy:{test}")
    print(classification_report(ytest,ypred))
    
    return model


# # knn 

# In[27]:


knn=mymodel(KNeighborsClassifier())


# # LogisticRegression

# In[28]:


logreg=mymodel(LogisticRegression())


# # SVM

# In[35]:


svm=mymodel(SVC(kernel = 'rbf'))


# # Decisiontree

# In[36]:


Decisiontree=mymodel(DecisionTreeClassifier
(criterion = 'entropy'))


# # randamforest

# In[33]:


randamforest=mymodel(RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0))


# # Accuracy Score of Support vector classifier : 0.97
# 
# From the accuracy and metrics above, the model that performed the best on the test data was the Support vector Classifier with an accuracy score of about 0.97 So let’s choose that model to detect cancer cells in patients. Make the prediction/classification on the test data and show both the Support vector Classifier model classification/prediction and the actual values of the patient that shows rather or not they have cancer.
# 
# And yay! we have successfully completed our Machine learning project.
