# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:11:34 2021

@author: Nabil Amimer
"""

import pandas as pd
from DecisionTree import DecisionTree as dt

'''
method that defines whether a variable is categorical or continuous.

keep in mind that it does so by looking at the number of possible values within
that category. 

you can always change the type of variable in the dataTypeClassifier 
'''
def defineCategoricalOrContinousCovariate(columnName):
    #initialize a set of the data in a column
    datasInColumns = set(dataFrame[columnName])
    #see if there are more than 10 different values
    
    #yes : continuous, no : categorical 
    if len(datasInColumns) > 10 :
        #continous
        return 1
    else:
        #categorical
        return 0
    
        
def defineTypeForEveryAttr():
    for columns in dataFrame:
        type = defineCategoricalOrContinousCovariate(columns)
        dataTypeClassifier[columns] = type 
    
    

#creation of the dataFrame    
dataFrame = pd.read_csv("processed.cleveland.data")
dataFrame["num"] = dataFrame["num"].replace([2 , 3 , 4] , 1)


#a list which contains the type of attribute present in our dataframe. 
'''
    this list only contains whether a columns/covariate
    contains categorical or continous variables
    0 : categorical || 1 : continuous
'''
dataTypeClassifier = {}
#immediatly fill the datatype Classifier
defineTypeForEveryAttr()

class RandomForest:

    def __init__(self, numberOfTrees , pandasDataFrame, attrToPredict):        
        self.numberOfTrees = numberOfTrees
        self.dataSet = pandasDataFrame
        
        self.trees = []
        


decisionTree = dt(dataFrame, dataTypeClassifier, "num")


