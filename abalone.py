# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 19:26:42 2022

@author: Nabil Amimer
"""

import pandas as pd
from RandomForest import RandomForest
import random
import time

start = time.time()

def defineCategoricalOrContinousCovariate(columnName, dataSet):
    """
    method that defines whether a variable is categorical or continuous.
    
    keep in mind that it does so by looking at the number of possible values within
    that category. 
    
    you can always change the type of variable in the dataTypeClassifier 
    """
    
    #initialize a set of the data in a column
    datasInColumns = set(dataSet[columnName])
    #see if there are more than 10 different values
    
    #yes : continuous, no : categorical 
    if len(datasInColumns) > 10 :
        #continous
        return 1
    else:
        #categorical
        return 0
    
        

def defineTypeForEveryAttr(dataSet, dataTypeClassifier):
    """
    Simple loop to go through all the covariates in the dataSet/dataFrame and 
    determine their type. 
    """
    for columns in dataSet:
        type = defineCategoricalOrContinousCovariate(columns, dataFrame)
        dataTypeClassifier[columns] = type 


#creation of the pandas dataFrame
dataFrame = pd.read_csv("abalone.data")

#a list which contains the type of attribute present in our dataframe. 
'''
    this list only contains whether a columns/covariate
    contains categorical or continous variables
    0 : categorical || 1 : continuous
'''
dataTypeClassifier = {}

#immediatly fill the datatype Classifier
defineTypeForEveryAttr(dataFrame, dataTypeClassifier)

#manually change "rings to categorical
dataTypeClassifier["rings"] = 0


#splitting of the dataframe into a training and testing set. 
dataFrameTrain = dataFrame.iloc[:4000].copy()
dataFrameTest = dataFrame.iloc[4000:].copy()


#to test the random froest, we can pick a random index from the test set and 
#run the "prediction()" method on it to retrieve "num". 
randomIndex = random.choice(dataFrameTest.index)
member = dataFrameTest.loc[randomIndex]

#creation of the random forest with 50 trees only. 
rf = RandomForest(dataFrameTrain, dataTypeClassifier, 50, "rings")    

#print the test results
print(randomIndex)
print(dataFrame.loc[randomIndex])
print(rf.predict(member))
end = time.time()

#print compilation time. 
print("compilation time : " + str(end - start)) 



