
import numpy as np

import pandas as pd

import matplotlib.pyplot as plt 

import seaborn as sns

plt.style.use('ggplot')

df = pd.read_csv('Iris.csv')

df

df.describe()

df.plot.scatter(x='SepalLengthCm',y='SepalWidthCm')

plt.show()

sns.FacetGrid(df, 
    hue="Species").map(plt.scatter, "SepalLengthCm", "SepalWidthCm").add_legend()
plt.show()

# TO CONVERT STRINGS TO LEBELS
labels = np.asarray(df.Species)
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
le.fit(labels)
labels = le.transform(labels)
labels

# to drop not relevant data
df_select = df.drop(['Id','SepalLengthCm','SepalWidthCm','Species'],axis =1)

# convert to dictionary, since we are using more than one column theredore we cant use np.asarray
df_features = df_select.to_dict(orient = 'record')

df_features

# need to convert this directory to vector for operations 
from sklearn.feature_extraction import DictVectorizer
vec = DictVectorizer()
features = vec.fit_transform(df_features).toarray()

features

#splitting data into test and training sets
from sklearn.model_selection import train_test_split

features_train, features_test, labels_train, labels_test = train_test_split(
    features, labels, 
    test_size=0.20, random_state=42)

# applying different classifiers
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVR
cfl1 = RandomForestClassifier()
cfl2 = SVR()

cfl1.fit(features_train,labels_train)

acc_test = cfl1.score(features_train,labels_train)

acc_test

cfl2.fit(features_train,labels_train)

cfl2.score(features_train,labels_train)

cfl1.score(features_test,labels_test)

cfl2.score(features_test,labels_test)

#predicting a type
flower = [[5.7,0.5]]
class_code = cfl1.predict(flower)
decoded_class = le.inverse_transform(class_code)
print (decoded_class)

class_code

