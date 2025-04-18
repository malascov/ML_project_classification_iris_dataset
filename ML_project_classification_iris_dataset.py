#!/usr/bin/env python
# coding: utf-8

# In[15]:


#import des libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import pandas as pd


# In[16]:


# import du dataset iris.data
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset=pd.read_csv((url), names=names)
dataset


# In[19]:


# shape
print(dataset.shape)


# In[21]:


# visualisation du head de nos donnees
print(dataset.head(20))


# In[24]:


print(dataset.describe())


# In[25]:


# la distribution des categories des fleurs dans les donnees qu'on a 
print(dataset.groupby('class').size())


# In[26]:


# boxplots
dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
pyplot.show()


# In[27]:


# histogrammes
dataset.hist()
pyplot.show()


# In[28]:


# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()


# In[29]:


#train_test split
array = dataset.values
X = array[:,0:4]
Y = array[:,4]
validation_size = 0.20
seed = 7
X_train, X_validation, Y_train, Y_validation = train_test_split(X, Y,
test_size=validation_size, random_state=seed)


# In[30]:


#ajouter les algos qu'on va essayer
models = []
models.append(('LR', LogisticRegression()))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC()))


# In[33]:


# tester l'accuracy de chaque modele
results = []
names = []
for name, model in models:
    kfold = KFold(n_splits=10, shuffle=True, random_state=seed)
    cv_results = cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# In[38]:


# Comparer les Algos
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show() 


# In[39]:


#on peut voir que KNN est le meilleur modele pour notre dataset


# In[40]:


# Make predictions on validation dataset
knn = KNeighborsClassifier()
knn.fit(X_train, Y_train)
predictions = knn.predict(X_validation)
print(accuracy_score(Y_validation, predictions))
print(confusion_matrix(Y_validation, predictions))
print(classification_report(Y_validation, predictions))


# In[ ]:


# accuracy = 90% ce qui est pas mal
# meme a partir de notre matrice de confusion on peut voir ou notre modele a fait des erreurs
# on peut realiser que y'a pas d'erreur au niveau de setosa ce qui est genial

