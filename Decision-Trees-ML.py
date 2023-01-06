# Author Dr. M. Alwarawrah
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

import math, os, time, scipy
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn import (linear_model ,preprocessing,metrics)
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
import sklearn.tree as tree

# start recording time
t_initial = time.time()

#Columns names
col_names = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K','Drug']
#Read dataframe and skip first raw that contain header
# data from https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-ML0101EN-SkillsNetwork/labs/Module%203/data/drug200.csv
df = pd.read_csv('drug200.csv',names=col_names, header = None, skiprows = 1)

#print Dataframe information
#print(df.describe())

#draw histograms for the following features
plt.clf()
df.hist()
plt.tight_layout()
plt.savefig("hist.png")

output_file = open('Decision_Trees_output.txt','w')

#define new data frame for all columns except 'Drug' and only take their values
X = df[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']] .values  

# pre-process data and convert categories to numerical values
# sex: F or M
X_sex = preprocessing.LabelEncoder()
X_sex.fit(['F','M'])
X[:,1] = X_sex.transform(X[:,1]) 

#BP LOW or NORMAL or HIGH
X_BP = preprocessing.LabelEncoder()
X_BP.fit([ 'LOW', 'NORMAL', 'HIGH'])
X[:,2] = X_BP.transform(X[:,2])

# Cholesterol: NORMAL  or HIGH
X_Chol = preprocessing.LabelEncoder()
X_Chol.fit([ 'NORMAL', 'HIGH'])
X[:,3] = X_Chol.transform(X[:,3]) 

#define new data frame for 'custcat' values
Y = df['Drug']

#define train and test  and set the test size to 0.3 and random_state to 3
X_train, X_test, Y_train, Y_test = train_test_split( X, Y, test_size=0.3, random_state=3)

# print the shape of each set:
print("X Train set dim: {} and Y Train set dim: {}".format(X_train.shape, Y_train.shape), file=output_file)
print("X Test set dim: {} and Y Test set dim: {}".format(X_test.shape, Y_test.shape), file=output_file)

#Model using Decision Tree Classifier and use entropy as your criterion {“gini”, “entropy”, “log_loss”} 
# with max depth of 4
drug_Tree = DecisionTreeClassifier(criterion="entropy", max_depth = 4)
drug_Tree
#fit the output using X_train and Y_train
drug_Tree.fit(X_train,Y_train)

# find the prediction Tree using X_test set
prediction_Tree = drug_Tree.predict(X_test)

#print the Decision Tree Accuracy 
print("Decision Trees's Accuracy: %.2f"%metrics.accuracy_score(Y_test, prediction_Tree), file=output_file)
#plot Drug Tree
plt.clf()
tree.plot_tree(drug_Tree)
plt.savefig('drug_Tree.png')

output_file.close()

#End recording time
t_final = time.time()
t_elapsed = t_final - t_initial
hour = int(t_elapsed/(60.0*60.0))
minute = int(t_elapsed%(60.0*60.0)/(60.0))
second = t_elapsed%(60.0)
print("%d h: %d min: %f s"%(hour,minute,second))