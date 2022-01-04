# -*- coding: utf-8 -*-
"""
Created on Sun Jan  2 12:55:59 2022

@author: Nabil Amimer
"""

from DecisionTree import DecisionTree
from collections import Counter

class RandomForest:
    """
    The random forest class will take the previous work done in this project 
    (Node, decisionTree) and combine them all together to create a randomForest. 
    
    Attributes:
    -----------
        
    numberOfTree : int, represents the # of trees. 

    dataSet : pandas.DataFrame, represents the dataSet used to create the training and testing sets for the trees. 

    trees : List, a list of all the trees in the forest. 

    attrToPredict : string, string that represents the name of the column that we are trying to predict. 

    trainingProportion : int, represents the # of elements that will go into the training set. 

    typeOfTree : int, boolean integer that represents Categorical || regression tree. 
    
    """

    def __init__(self, pandasDataFrame, numberOfTrees, attrToPredict , trainingProportion=0.66):        
        
        self.numberOfTrees = numberOfTrees
        self.dataSet = pandasDataFrame
        self.trees = []
        self.attrToPredict = attrToPredict
        self.trainingProportion = trainingProportion
        
        
        #a list which contains the type of attribute present in our dataframe. 
        '''
            this list only contains whether a columns/covariate
            contains categorical or continous variables
            0 : categorical || 1 : continuous
        '''
        self.dataTypeClassifier = {}
        
        #immediatly fill the datatype Classifier
        self.defineTypeForEveryAttr()
        
        self.typeOfTree = self.dataTypeClassifier[attrToPredict]
        
        #filling the trees of the randomForest
        self.createRandomForestTrees()
        
    
    def createRandomForestTrees(self):
        for i in range(self.numberOfTrees):
            self.trees.append(DecisionTree(self.dataSet, self.dataTypeClassifier,
                                    self.attrToPredict, self.trainingProportion))
            print("tree " + str(i) + " created!")
    
    def OOBprediction(self, index):
        predictions = {}
        member = self.dataSet.loc[index]
        
        for tree in self.trees:
            #look if the index is out of bag for this tree
            if index in tree.testIndexList:
                treePrediction = tree.goThroughNodes(tree.mainNode, member)
                predictions[tree.index] = treePrediction[self.attrToPredict]
                print("this is tree #" + str(tree.index) + " and its predictiong is " + str(treePrediction["num"]))
        
        
        
        if self.typeOfTree == 0:
            #look for majority 
            prediction = self.getMostRepeatedElementInDict(predictions)
            return prediction
        else:
            partialSum = 0
            for key in predictions:
                partialSum += predictions[key]
                
            average = partialSum/len(predictions)
            return average
            
        
    
    def getMostRepeatedElementInDict(self, dictionnary):
        elements = []
        for key in dictionnary:
            elements.append(dictionnary[key])
            
        c = Counter(elements)
        s = c.most_common(1)[0]
        return s
            
    
    def defineCategoricalOrContinousCovariate(self, columnName):
        """
        method that defines whether a variable is categorical or continuous.
        
        keep in mind that it does so by looking at the number of possible values within
        that category. 
        
        you can always change the type of variable in the dataTypeClassifier 
        """
        
        #initialize a set of the data in a column
        datasInColumns = set(self.dataSet[columnName])
        #see if there are more than 10 different values
        
        #yes : continuous, no : categorical 
        if len(datasInColumns) > 10 :
            #continous
            return 1
        else:
            #categorical
            return 0
        
            

    def defineTypeForEveryAttr(self):
        """
        Simple loop to go through all the covariates in the dataSet/dataFrame and 
        determine their type. 
        """
        for columns in self.dataSet:
            type = self.defineCategoricalOrContinousCovariate(columns)
            self.dataTypeClassifier[columns] = type 
    
        
        
        
        
        
        
        
        