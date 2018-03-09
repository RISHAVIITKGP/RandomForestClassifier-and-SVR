
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import pandas as pd


# In[3]:


import matplotlib.pyplot as plt 


# In[4]:


import seaborn as sns


# In[5]:


plt.style.use('ggplot')


# In[6]:


df = pd.read_csv('Iris.csv')


# In[7]:


df


# In[8]:


df.describe()


# In[9]:


df.plot.scatter(x='SepalLengthCm',y='SepalWidthCm')


# In[10]:


plt.show()


# In[11]:


sns.FacetGrid(df, 
    hue="Species").map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
plt.show()


# In[12]:


# TO CONVERT STRINGS TO LEBELS
labels = np.asarray(df.Species)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
labels


# In[13]:


# to drop not relevant data
df_select = df.drop(['Id','SepalLengthCm','SepalWidthCm','Species'],axis =1)


# In[15]:


# convert to dictionary, since we are using more than one column theredore we cant use np.asarray
df_features = df_select.to_dict(orient = 'record')


# In[21]:


df_features


# In[16]:


# need to convert this directory to vector for operations 
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
features = vec.fit_transform(df_features).toarray()


# In[17]:


features


# In[18]:


#splitting data into test and training sets
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, 
    test_size=0.20, random_state=42)


# In[19]:


# applying different classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
cfl1 = RandomForestClassifier()
cfl2 = SVR()


# In[20]:


cfl1.fit(features_train,labels_train)


# In[21]:


acc_test = cfl1.score(features_train,labels_train)


# In[22]:


acc_test


# In[23]:


cfl2.fit(features_train,labels_train)


# In[24]:


cfl2.score(features_train,labels_train)


# In[25]:


cfl1.score(features_test,labels_test)


# In[26]:


cfl2.score(features_test,labels_test)


# In[27]:


#predicting a type
flower = [[5.7,0.5]]
class_code = cfl1.predict(flower)
decoded_class = le.inverse_transform(class_code)
print (decoded_class)


# In[28]:


class_code

