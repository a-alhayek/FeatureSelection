#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Feb  6 22:25:03 2020

@author: ahmadalhayek
"""

# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#import statsmodels.regression.linear_model as sm
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
# reading the file and creating a metrix out of it
data = pd.read_csv('50_Startups.csv') 
#print type(data) # data is type <class 'pandas.core.frame.DataFrame'>
# Extracting Independent and dependent variables
X = data.iloc[:, :-1].values #iloc transform the data to array
#print type(X)
Y = data.iloc[:,-1].values

# converting categorical data to dummy varibles
labelEncoder_X = LabelEncoder()

#print X[:,3] before label endocder the values were names
X[:,3] = labelEncoder_X.fit_transform(X[:,3])
#print X[:,3] now values are numbers
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer(
    [('one_hot_encoder', OneHotEncoder(), [3])],    # The column numbers to be transformed (here is [0] but can be [0, 1, 3])
    remainder='passthrough'                         # Leave the rest of the columns untouched
)
#print type(ct) # type <class 'sklearn.compose._column_transformer.ColumnTransformer'>

#x = np.array(ct.fit_transform(x), dtype=np.float)


#oneHotEncoder = OneHotEncoder(categorical_features = [3])
X = np.array(ct.fit_transform(X), dtype=np.float) # changeing it back to array

X =X[:,1:]
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2)

# train_Test_solit is a class the return 4 virable in this order


regression = LinearRegression()
regression.fit(X_train,Y_train)

# fit is to train your data

#predict the Test results

Y_pred = regression.predict(X_test)

#Building an Optimal Model using backward Elimination
X = np.append(arr = np.ones((50,1), dtype = int) , values =X ,axis=1)
#Declare an optimal matrix of features .
X_opt = X[:,:]

regression_OLS = sm.OLS(endog = Y,exog = X_opt).fit()
#print(regression_OLS.summary())
#elementing the highest p value\
X_opt = X[:,[0,1,3,4,5]]
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#print(regression_OLS.summary())
X_opt = X[:,[0,3,4,5]]
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#print(regression_OLS.summary())
X_opt = X[:,[0,3,5]]
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
#print(regression_OLS.summary())

X_opt = X[:,3]
regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()

regression_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
ar_x = np.zeros((50,6))
#ar_x[:,3] = X[:,3]
#print X[:,1]
ar_x[:,0] = X[:,1]
#print ar_x[:,0]

regression_OLS = sm.OLS(endog = Y, exog = ar_x).fit()

#print min(regression_OLS.pvalues)
#print (min(regression_OLS.pvalues) < 0.05)
#print (len(X[0]))


def forwardSelection(x,y,sigLevel): # take in an array of features and array of targets
    ar_x = np.zeros((len(x),len(x[0]))) #creating an array of zeros with the same amount of rows and columns as the data X
    numVar = len(X[0])  # the initial size of the data columns
    in_x = 0 # the moving index of ar_x points to the first column in ar_x
    totalFeature = 0 # the amount of features we are keeping
    
    for i in range (0,numVar):    # array start from 0 to data column's length
        ar_z = np.zeros(numVar-i) # creating an array to store the pValues of the data columns
        for j in range(0, numVar-i):   # substracting the number of the features we select 
            ar_x[:,in_x] = x[:,j]    # moving index take the values of the x(data) column      
            regression_OLS = sm.OLS(Y, ar_x).fit() # fitting our model
           # print regression_OLS.summary() 
            ar_z[j] = regression_OLS.pvalues[i] # storing the pvalues in array
        min_p =  min(ar_z)# taking the smallest value in my data pValues 
       # print ar_z
        if min_p <sigLevel: # if the smallest value less than sigLevel 
            in_p = np.where(ar_z == min_p) #take the index of the pValue  
       #print in_p[-1][0]
            ar_x[:,in_x] = x[:,in_p[0][0]] #take x's column and store it to in array 
            in_x = in_x +1# move the index after storing the data
            x = np.delete(x,in_p[0][0],1)# delete the data from x
            totalFeature = totalFeature +1# incressing
        else:
            ar_x = np.delete(ar_x,in_x,1)
            break
    row_delete = numVar - totalFeature -1
    for i in range (0,row_delete):
        ar_x = np.delete(ar_x,totalFeature,1)    
    regression_OLS = sm.OLS(Y, ar_x).fit()
    print regression_OLS.summary() 
    return ar_x
    
            
     
def forwardSelection(x,y,sigLevel):
    
    ar_x = np.zeros((len(x),1))#array of size one column of our data
    ar = np.zeros((len(x),1))#array of zeros of size one column to fit in out data 
    numvar = len(x[0]) # an integer to save data features length 
    ar_features_removed = np.zeros((len(x[0])))
    for i in range (0,numvar):
        ar_z = np.zeros(numvar-i)
        
        for j in range(0,numvar-i):
            ar_x[:,i] = x[:,j]
            regression_OLS = sm.OLS(y,ar_x).fit()
            ar_z[j] = regression_OLS.pvalues[i]
           
           
        min_p = min(ar_z)
        if min_p < sigLevel:
            in_p = np.where(ar_z == min_p)
            ar_features_removed[i] = in_p[0][0]
            ar_x[:,i] = x[:,in_p[0][0]]
            x = np.delete(x,in_p[0][0],1)
            ar_x = np.append(ar_x,ar,1)
        else:
            ar_x = np.delete(ar_x,i,1)
            break
    
    regression_OLS = sm.OLS(y,ar_x).fit()
    print "This is C)forward Selection OLS Regression Results  "
    print(regression_OLS.summary()) # printing all the data
    print " "
    print " "

    print 'p-values for c) forward Selection'
    array_pvalues = regression_OLS.pvalues
    for i in range(0, len(array_pvalues)):
        print 'p-value for feature ',i,' is ',array_pvalues[i]
    print "featured deleted are"
    print ar_features_removed

    return ar_x
        
    





def backwardElimination(X,Y, sigLevel): 
    
    
    numVar = len(X[0]) 
    print numVar
    for i in range(0, numVar): 
        
        regression_OLS = sm.OLS(Y, X).fit()    
        maxVar = max(regression_OLS.pvalues).astype(float) 
        print regression_OLS.summary() 
        if maxVar > sigLevel:       
            
            for j in range(0, numVar - i):      
                if(regression_OLS.pvalues[j].astype(float)==maxVar): 
                    print (regression_OLS.summary())
                    X = np.delete(X,j,1)   
                    print X
    print regression_OLS.summary() 
    

    return X





X = np.delete(X,0,1)
print X[0]
forwardSelection(X,Y,0.05)