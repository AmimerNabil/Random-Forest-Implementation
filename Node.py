# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 18:16:17 2021

@author: Nabil Amimer
"""

'''
This class represents each Node that our decision tree will have

Trying to keep this as simple as possible, we will leave it to only a few 
instance variables

1. a list of its branches (Link to other nodes)
2. the split decided at this node. 
3. the population at this node
4. and the prediction at that node if it is a terminal node
'''
class Node:
    
    counter = 0
    
    def __init__(self):
        
        #A dictionnary of the population indexes and the related outcome of the aimed attr 
        self.populationAtNode = {}
        
        #remaining Splits in this node
        self.remainingSplits = []

        #prediction at this node 
        '''
        made into a dictionnary as it will have the attribute we are trying to predict as a key and
        the value of the prediction at that key. 
        '''
        self.predictionAtThisNode = {}
        
        #bestSplit gining the node
        self.bestCovariate = ""
        self.split = {}

        #branches and link to other ndoes
        self.branches = {}
        
        #boolean that represents whether this node is terminal
        self.isTerminal = False
        
        #simple index for the node
        Node.counter += 1
        self.index = Node.counter
        
        #special index representing the terminal node index. 
        #if value of TIndex = -1, it means that it is not a terminal node. 
        self.TIndex = -1
        
        
    '''
    this method is used to get the prediction at the current node. 
    
    it splits the prediction calculation in two possibilities :
        
        1. regression tree 
        
            returns the average of all the population in the tree
        
        2. classification tree
        
            returns the class with the msot population in it. 
    '''
    def getPrediction(self, typeOfTree, attrToPredict, dataSet):
        
        #regression
        if typeOfTree == 1:
            partialSum = 0
            for member in self.populationAtNode:
                partialSum += dataSet.loc[member][attrToPredict]
                
            average = partialSum/len(self.populationAtNode)
            return average
        
        #classification
        else : 
            categories = set(dataSet[attrToPredict])
            
            majorityDict = {}
            
            for element in categories:
                majorityDict[element] = 0
            
            for member in self.populationAtNode:
                majorityDict[dataSet.loc[member][attrToPredict]] += 1
            
            
            return max(majorityDict, key=majorityDict.get)
        
        
        
        
        
        
        
        