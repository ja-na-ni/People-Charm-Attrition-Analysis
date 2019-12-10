# -*- coding: utf-8 -*-
"""
Created on Thu Nov 14 12:12:22 2019

@author: win10
"""
DECISION TREE FOR PEOPLE CHARM DATA
---------------------------------------------------------------------------------------------
import pandas as pd
data = pd.read_excel('People Charm case.xls')

x = data.drop(['Left','WorkAccident'],axis=1)
y = data.loc[:,'Left']

x = pd.get_dummies(data,drop_first=True) 

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
x = sc.fit_transform(x)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25, random_state = 0)

from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = 'entropy')
classifier.fit(x_train,y_train)

y_pred = classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
print(cm)

accuracy = (cm[0,0]+cm[1,1])/(cm[0,0]+cm[1,1]+cm[0,1]+cm[1,0])

print(accuracy*100)

classifier.score(x_test,y_test)

import pydotplus
from sklearn.tree import export_graphviz

dot = export_graphviz(classifier, out_file= None,filled = True,rounded = True)
graph = pydotplus.graph_from_dot_data(dot)
graph.write_png('charmDT.org')