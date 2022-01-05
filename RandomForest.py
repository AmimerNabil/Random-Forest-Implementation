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

    def __init__(self, pandasDataFrame, DTC, numberOfTrees, attrToPredict , trainingProportion=0.66):        
        
        self.numberOfTrees = numberOfTrees
        self.dataSet = pandasDataFrame
        self.trees = []
        self.attrToPredict = attrToPredict
        self.trainingProportion = trainingProportion
        self.dataTypeClassifier = DTC
        
        self.typeOfTree = self.dataTypeClassifier[attrToPredict]
        
        #filling the trees of the randomForest
        self.createRandomForestTrees()
        
    
    def createRandomForestTrees(self):
        """
        Creates the trees of the randomForest. 
        parameters
        ----------
        None
        
        Returns
        -------
        None
        
        """
        print("<------ Creation of the random forest ------>")
        
        print("making the trees...")
        for i in range(self.numberOfTrees):
            self.trees.append(DecisionTree(self.dataSet, self.dataTypeClassifier,
                                    self.attrToPredict, self.trainingProportion))
            print("tree " + str(i) + " created!")
            
        print("<------ Random Forest Created! ------>\n")
    
    
    def predict(self, member):
        """
        makes a prediction based on an indivual or an element that is not in the dataSet of the random forest.
        (a completely new value for the forest). It makes the prediciton using all the trees in the forest as none of the trees have it 
        in their trainset. 
        
        Parameters
        ----------
        member : pandas.Series
            an element of a dataFrame that is in no training set or test set of any of the trees in the forest. 
            

        Returns
        -------
        prediction : dictionnary
            the majority class or the average the terminal nodes. 

        """
        
        
        print("<----- begining of a prediction ----->")
        
        predictions = {}
        
        print("\t\tgoing through the trees...")
        for tree in self.trees:
            treePrediction = tree.goThroughNodes(tree.mainNode, member)
            predictions[tree.index] = treePrediction[self.attrToPredict]
            print("this is tree #" + str(tree.index) + " and its prediction is " + str(treePrediction[self.attrToPredict]))
    
    
        
        finalPrediction = self.returnPrediction(predictions)
        
        print(" <-------- ACTUAL MEMBER AND ITS ATTRIBUTES--------->")
        print(member)
        print(" <-------- PREDICTION --------->")
        print("prediction for " + self.attrToPredict + " = " + str(finalPrediction))
        print("<-------- end --------->")
        return finalPrediction
            
    
    def OOBprediction(self, index):
        """
        OOB prediction is a prediction that is done using a member of the data
        Set that is already in the forest dataSet. However we make predictions using 
        only the trees that have the member in their Out of bag space. (test set)
        
        Parameters
        ----------
        
        index : the index of the individual we are trying to make a prediciton out of.
        
        return
        ------
        
        prediction : a dictionary with the prediction and the # of trees that have the same prediction. 
        
        """
        predictions = {}
        member = self.dataSet.loc[index]
        
        for tree in self.trees:
            #look if the index is out of bag for this tree
            if index in tree.testIndexList:
                treePrediction = tree.goThroughNodes(tree.mainNode, member)
                predictions[tree.index] = treePrediction[self.attrToPredict]
                print("this is tree #" + str(tree.index) + " and its predictiong is " + str(treePrediction["num"]))
        
        return self.returnPrediction(predictions)
            
        
    
    def getMostRepeatedElementInDict(self, dictionnary):
        """
        this method is to get the most repeated elements in a dictionary.
       
       Return
       ------
       returns the value that appears the most. 
       
        """
        elements = []
        for key in dictionnary:
            elements.append(dictionnary[key])
            
        c = Counter(elements)
        s = c.most_common(1)[0]
        return s
            
        
    def returnPrediction(self, predictionDict):
        """
        returns a the final prediction based on the prediction dictionnary
        
        Parameters
        ----------
        
        predictionDict : dict
            dictionary that contains the prediction of all trees involved. 
            
        Returns
        -------
        
        returns the prediciton that suits the type of tree.
        """
        if self.typeOfTree == 0:
            #look for majority 
            prediction = self.getMostRepeatedElementInDict(predictionDict)
            return prediction[0]
        
        else:
            partialSum = 0
            for key in predictionDict:
                partialSum += predictionDict[key]
                
            average = partialSum/len(predictionDict)
            return average
        
        
        
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
    
        

def defineDataTypeClassifier(dataFrame, dataTypeClassifier):
    """
    Simple loop to go through all the covariates in the dataSet/dataFrame and 
    determine their type. 
    """
    for columns in dataFrame:
        type = defineCategoricalOrContinousCovariate(columns, dataFrame)
        dataTypeClassifier[columns] = type 
   
        