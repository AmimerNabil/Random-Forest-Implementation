# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 17:58:48 2021

@author: Nabil Amimer
"""
import random
from Node import Node


class DecisionTree:    
    
    #class variable used to keep track of the trees
    counter = 0
    '''
    this class represents the decision tree class. 
    
    the original size of the training set is set at 66% but can be modified at will
    
    @parameters :
        dataSet - a pandas DataFrame
        attrToPredict - the name of the attribute you are trying to predict in your dataFrame
        trainingProportion - the percentage of the data set that you wish to use as a trainingSet
    '''
    def __init__(self , dataSet, DTC, attrToPredict, trainingProportion=0.66):
        
        #initialization of the basic instance variables. 
        self.dataSet = dataSet #data frame containing all the data
        self.numberOfData = len(self.dataSet)
        self.trainingQty = int(self.numberOfData*trainingProportion)
        self.testQty = self.numberOfData - self.trainingQty
        self.attrToPredict = attrToPredict
        
        #creation of the trainingSet and TestSet
        #only making a list of the indexes, not the whole members. 
        self.indexList = [i for i in range(self.numberOfData)]
        self.trainingIndexList = random.choices(self.indexList, k = self.trainingQty)
        self.testIndexList = list(set(self.indexList).difference(set(self.trainingIndexList)))
        
       
        self.testIndexList.sort()
        self.trainingIndexList.sort()
        
        #reference to the dataTypeClassifier in main. 
        self.dataTypeClassifier = DTC
        
        
        #this is the type of tree depending on the attr we are trying to predict. 
        self.typeOfTree = self.dataTypeClassifier[self.attrToPredict]
        
        
        self.mainNode = Node()
        self.initializeMainNode(self.mainNode)
        self.getBestPossibleSplitFromPossibleSplits(self.mainNode)
        
        self.terminalNodes = 0
        
        self.createBranches(self.mainNode)
        
        self.index = DecisionTree.counter
        DecisionTree.counter += 1
        
    
    def goThroughNodes(self, node, index):
        
        #1st -> see if this node is a terminal Node
        if node.isTerminal:
            prediction = node.predictionAtThisNode
            return prediction
            
        #if it is not a terminal node, find the category where our member belongs and recursively look into this node
        else:
            split = node.bestCovariate
            member = self.dataSet.loc[index]
            
            memberValueForCovariate = member[split]
            
            
            if memberValueForCovariate == "?":
                branch = self.getNodeWithMostPeople(node)  
                return self.goThroughNodes(node.branches[branch], index)
        
            #if continuous
            if self.dataTypeClassifier[split] == 1:
                if memberValueForCovariate >= node.split[split][0]:
                    return self.goThroughNodes(node.branches["left"], index)
                else:
                    return self.goThroughNodes(node.branches["right"] , index)
            
            #categorical
            else:
                for branch in node.branches:
                    #if the member value = the name of the branch, it is part of it. 
                    if memberValueForCovariate == branch:
                        return self.goThroughNodes(node.branches[memberValueForCovariate] , index)   
            
                        
    '''
    we initialize the main node by defining the best split and its corresponding 
    prediction as well as its remaining splits. we will then use the recursion function
    "createBranch()" to compute the rest of the tree. 
    '''
    def initializeMainNode(self , node):
        #population at the node
        node.populationAtNode = self.trainingIndexList
        
        #initialize prediction
        node.predictionAtThisNode[self.attrToPredict] = node.getPrediction(self.typeOfTree, self.attrToPredict, self.dataSet)
        
        #remainingSplits
        for possibleSplit in self.dataSet:
            if possibleSplit != self.attrToPredict:
                node.remainingSplits.append(possibleSplit)
            
            
    '''
    this method creates the branches for a specific node. 
    It is a recursive function and the stopping criterion used is 
    the size of the population in the node. 
    
    -> if more than 30, populate branches
    -> else stop here and make this a terminal node. 
    
    '''
    def createBranches(self, node, StoppingCriterionPopulation = 30):
        #attribute used for the split in this node
        nodeAttrSplit = node.bestCovariate

        #if the data is continuous
        if self.dataTypeClassifier[nodeAttrSplit] == 1:
            
            #as there are only left and right branches possible with continuous branches
            node.branches["left"] = Node()
            node.branches["left"].remainingSplits = list(node.remainingSplits)
            node.branches["left"].nodeCharacteristics = dict(node.nodeCharacteristics)
            
            node.branches["right"] = Node()
            node.branches["right"].remainingSplits = list(node.remainingSplits)
            node.branches["right"].nodeCharacteristics = dict(node.nodeCharacteristics)
            
            
            #populate the population at each branch with the corresponding indexes.
            for index in node.populationAtNode:
                member = self.dataSet.loc[index]
                if member[nodeAttrSplit] >= node.split[nodeAttrSplit][0]:
                    
                    node.branches["left"].populationAtNode[index] = self.dataSet.loc[index][self.attrToPredict]
                    node.branches["left"].nodeCharacteristics[nodeAttrSplit] = "bigger than"
                else:
                    node.branches["right"].populationAtNode[index] = self.dataSet.loc[index][self.attrToPredict]
                    node.branches["right"].nodeCharacteristics[nodeAttrSplit] = "smaller than"
            
            
        #if the data is categorical
        else :
            #define all the possible values 
            possibleValues = set(self.dataSet[nodeAttrSplit])    
            
            for values in possibleValues:
                if values != "?":
                    #create the node for each branch
                    node.branches[values] = Node()
                    
                    #give it its remaining splits from the node
                    node.branches[values].remainingSplits = list(node.remainingSplits)
                    node.branches[values].nodeCharacteristics = dict(node.nodeCharacteristics)
                    node.branches[values].nodeCharacteristics[nodeAttrSplit] = values
            
            
            #populate the population at each branch with the corresponding indexes.
            for index in node.populationAtNode:
                member = self.dataSet.loc[index]
                valueAtCovariate = member[nodeAttrSplit]

                #if the value is unknown, used a simplistic approach where i 
                #send the index to the node with the most population. 
                if valueAtCovariate == "?":
                    keyWithMostPeople = self.getNodeWithMostPeople(node)      
                        
                    #adding that index into the branch population
                    node.branches[keyWithMostPeople].populationAtNode[index] = self.dataSet.loc[index][self.attrToPredict]
                else:
                    #proceed as usual if not an unkonwn value. 
                    node.branches[valueAtCovariate].populationAtNode[index] = self.dataSet.loc[index][self.attrToPredict]
        
        
        #recursive part of the function -> 
        for key in node.branches:
            
            '''
            in this for loop, we go through each branch of the main node and 
            initiate a recursive pattern where we initialize the sub trees. 
            
            we initialize the prediction at each node, the population, and whether it is a terminal node. 
            '''
            
            currentNode = node.branches[key]
            
            currentNode.predictionAtThisNode[self.attrToPredict] = currentNode.getPrediction(self.typeOfTree , self.attrToPredict , self.dataSet)

            #initialize the best split for the current node.
            self.getBestPossibleSplitFromPossibleSplits(currentNode)
            
            #verification of the stopping criterion.
            if len(currentNode.populationAtNode) >= StoppingCriterionPopulation:
               self.createBranches(currentNode)        
            else:
                #make node a terminal ndoe if stopping criterion is reached. 
                currentNode.isTerminal = True
                self.terminalNodes += 1
                currentNode.TIndex = self.terminalNodes
                
        
    
    def getNodeWithMostPeople(self, node):
        keyWithMostPeople = ""
        length = 0
        
        #getting the branch with the biggest population
        for key in node.branches:
            temp = len(node.branches[key].populationAtNode)
            keyWithMostPeople = key if temp > length else keyWithMostPeople
        
        return keyWithMostPeople
    
    
    #now make a method for the gini index
    def measureGiniForCovariate(self, covariate , numberOfDivisions = 5):
        
        type = self.dataTypeClassifier[covariate]
        
        possibleValues = set(self.dataSet[covariate])    
        
        #continuous
        if type == 1:
            maxValue = max(possibleValues)
            minValue = min(possibleValues) 
            
            #defining possible divisions within the covariate
            divisionSize = (maxValue - minValue)/numberOfDivisions
            splits = []
            
            
            for i in range(1 , numberOfDivisions - 1):
                splits.append(minValue + divisionSize * i)
            
            giniPerSplit = {}
                    
            for split in splits:
                #people who are yes from the split
                populationLeft = {}
                #people who are no from teh split
                populationRight = {}
                
                #divide the trainingIndex into left and right with respect to the split
                for index in self.trainingIndexList:
                    
                    member = self.dataSet.loc[index]
                    if member[covariate] > split:
                        populationLeft[index] = member[self.attrToPredict]
                    else:
                        populationRight[index] =  member[self.attrToPredict]
                        
                
                #calculate the gini index for the left and right brnach
                
                
                #left
                numberOfPeopleLeft = len(populationLeft)
                giniLeft = 0
                if numberOfPeopleLeft != 0:
                    giniLeft = self.giniAtNode(populationLeft , numberOfPeopleLeft)
                
                #right
                numberOfPeopleRight = len(populationRight)
                giniRight = 0
                if numberOfPeopleRight != 0:
                    giniRight = self.giniAtNode(populationRight , numberOfPeopleRight)
                
                #get total gini index for the covariate 
                GiniWithSplit = giniLeft + giniRight
                
                #add it to the possible splits within the continuous covariate
                giniPerSplit[split] = GiniWithSplit
            
            #return the best split
            bestSplitWithinCovariate = min(giniPerSplit , key=giniPerSplit.get)
            bestSplit = [bestSplitWithinCovariate , giniPerSplit[bestSplitWithinCovariate]]
            return bestSplit
        
        else:        
            #dictionnary that will contain a list of every member that has the same value for that covariate. 
            branches = {}
            
            #for this implementation of the random tree, we do not create seperate categories for missing values.
            for pValue in possibleValues:
                if pValue != "?":
                    branches[pValue] = {}
                        
            #now that we have the branches, place every member into its corresponding categories
            for index in self.trainingIndexList:
                #get the member from the dataSet
                member = self.dataSet.loc[index]
                
                #see the value of the meme
                if(member[covariate] == "?"):
                    branchWithMostMembers = self.getBranchWithMostElements(branches)
                    branches[branchWithMostMembers][index] = member[self.attrToPredict]
                else :
                    branches[member[covariate]][index] = member[self.attrToPredict]
            
                
            #adding the gini index at all branch
            giniIndex = 0
            
            for key in branches:
                population = branches.get(key)
                if len(population) != 0:
                    giniAtBranch = self.giniAtNode(population , len(population)) 
                    giniIndex += giniAtBranch
            
            #returning the gini
            return giniIndex
    
    
    '''
    Get the branch with the most element
    This method is only used for the measurement of the best possible split.
    '''
    def getBranchWithMostElements(self, dictBranch):
        keyWithMostElements = ""
        numberOfElements = 0;
        
        for key in dictBranch:
            if len(dictBranch[key]) > numberOfElements:
                keyWithMostElements = key
            
        return keyWithMostElements
    


    '''
    this method is used to get the gini index at a specific Node. 
    
    We calculate the Gini using the formula ->
    Gini at Node = 1 - (probability of yes)^2 - (probability of no)^2
    
    It is a weighted index and so we multiply it with the length of the population
    sample at the node. 
    '''
    def giniAtNode(self, population , numberOfPeopleAtNode):
        peopleYes = 0;
        for key in population:
            if population[key] != 0:
                peopleYes += 1
        
        peopleNo = numberOfPeopleAtNode - peopleYes
        probabilityYes = peopleYes/numberOfPeopleAtNode
        probabilityNo = peopleNo/numberOfPeopleAtNode
        
        gini = numberOfPeopleAtNode*(1 - (probabilityYes)**2 - probabilityNo**2)
        
        return gini
        
    
    '''
    this method is used to tri within the gini indexes in the remaining splits to 
    find what which one would be best for the population
    
    
    reduced Remaining split quantity is set at 80% of the available remainingSplits
    can be changed
    '''
    def getBestPossibleSplitFromPossibleSplits(self, node, percentage = 0.8):
        
        #list that will contain ginis
        giniWithCovariate = {}
        for split in node.remainingSplits:
            giniWithCovariate[split] = self.measureGiniForCovariate(split)
            
            
        #simple variable to keep track of smallest gini
        covariate = ""
        smallestGini = -1
        g = 0
        
        #picking at random from the availableSplits to add randomness. 
        newLength = int(percentage*len(giniWithCovariate))
        reducedRemainingSplits = random.sample(list(giniWithCovariate), k=newLength)
        
        #loop through the ginis
        for ginis in reducedRemainingSplits:
            
            '''
            accessing the ginis depends on the nature of the var.
            
            as they are placed into dictionaries, ginis are accessible through 
            their key however if it is a continuous RV, the key is linked to a 
            list with two elemetns :
                
                [0] -> best split within continuous values.
                [1] -> gini associated with the the split.
            
            
            '''
            if self.dataTypeClassifier[ginis] == 0:
                g = giniWithCovariate[ginis]
            else:
                g = giniWithCovariate[ginis][1]
               
                
            if smallestGini < 0: 
                covariate = ginis
                smallestGini = g
            elif g < smallestGini:
                covariate = ginis
                smallestGini = g
    
        
        node.remainingSplits.remove(covariate)
        node.split = {covariate : giniWithCovariate[covariate]}
        node.bestCovariate = covariate

        return node.split
    
    






