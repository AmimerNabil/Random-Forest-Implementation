# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:11:34 2021

@author: Nabil Amimer
"""

import pandas as pd
import random
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
sizeOfTestSet = len(decisionTree.testIndexList)

#test with the training index of the random tree
success = 0
for i in range(sizeOfTestSet):
    print("<--------------- new test ( " +str(i)+ ") ---------------->")
    randomTestIndex = int(random.choice(decisionTree.testIndexList))
   # print(randomTestIndex)
    member = dataFrame.loc[randomTestIndex]    
    actual = member["num"]
   # print(actual)
    prediction = decisionTree.goThroughNodes(decisionTree.mainNode, randomTestIndex)
    print(actual)
    print(prediction["num"])
    
   # print(prediction)
    if actual == prediction["num"]:
        success += 1
        

print("success = " + str(success))
print("failure = " + str(sizeOfTestSet - success))

    
