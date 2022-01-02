# -*- coding: utf-8 -*-
"""
Created on Thu Dec 30 17:58:48 2021

@author: Nabil Amimer
"""
import random
from Node import Node


class DecisionTree:  
    """
    Decision tree class. 
    """
    
    #class variable used to keep track of the trees
    counter = 0
    
    def __init__(self , dataSet, DTC, attrToPredict, trainingProportion=0.66):
        """
        this class represents the decision tree class. 
        the original size of the training set is set at 66% but can be modified at will
        
        @parameters :
            dataSet -> a pandas DataFrame
            
            attrToPredict -> the name of the attribute you are trying to predict in your dataFrame
            
            trainingProportion -> the percentage of the data set that you wish to use as a trainingSet
        """
        
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
        
    
    def goThroughNodes(self, node, member):
        """
            This method is used to make a prediction given the index of an individual. 
            If given the index of a member of the training set, it will give the right 
            value as the model was created with it. 
            
            However giving it the index value of a member in the test set (out of bag)
            will give you a prediction of his state. We can then compare the actual value 
            given and the prediction to measure the accuracy of our decision tree. 
       
            #params :
                node -> current node where we test its split in search for the terminal node.
                
                member -> individual or sample from the population for which we would like to konw
                the prediction. 
        """
        
        #1st -> see if this node is a terminal Node
        if node.isTerminal:
            prediction = node.predictionAtThisNode
            return prediction
            
        #if it is not a terminal node, find the category where our member belongs and recursively look into this node
        else:
            split = node.bestCovariate
          
            memberValueForCovariate = member[split]
            
            #same handling as when we measure the gini index, we simply go through the node that has the most population
            #there is a lot of place for improvement in the way we deal with unknown values. 
            if memberValueForCovariate == "?":
                branch = self.getNodeWithMostPeople(node)  
                return self.goThroughNodes(node.branches[branch], member)
        
            #if continuous
            if self.dataTypeClassifier[split] == 1:
                if memberValueForCovariate >= node.split[split][0]:
                    return self.goThroughNodes(node.branches["left"], member)
                else:
                    return self.goThroughNodes(node.branches["right"] , member)
            
            #categorical
            else:
                for branch in node.branches:
                    #if the member value = the name of the branch, it is part of it. 
                    if memberValueForCovariate == branch:
                        return self.goThroughNodes(node.branches[memberValueForCovariate] , member)   
            
                
            
    def initializeMainNode(self , node):
        """
        we initialize the main node by defining the best split and its corresponding 
        prediction as well as its remaining splits. we will then use the recursion function
        "createBranch()" to compute the rest of the tree. 
      
        @params:
            node -> main node that we initialize
        """
        #population at the node
        node.populationAtNode = self.trainingIndexList
        
        #initialize prediction
        node.predictionAtThisNode[self.attrToPredict] = node.getPrediction(self.typeOfTree, self.attrToPredict, self.dataSet)
        
        #remainingSplits
        for possibleSplit in self.dataSet:
            if possibleSplit != self.attrToPredict:
                node.remainingSplits.append(possibleSplit)
            
            
    def createbranches(self, node, StoppingCriterionPopulation = 30):
        """
        this method creates the branches for a specific node. 
        It is a recursive function and the stopping criterion used is 
        the size of the population in the node. 
        
        -> if more than 30, populate branches
        
        -> else stop here and make this a terminal node. 
        
        @params:
            node -> base node on which we determine the population of its branhces. 
        
            StoppingCriterionPopulation -> we use the length of the population of the node as 
            a stopping criterion. It is by default set at 30. 
        """
        #attribute used for the split in this node
        nodeAttrSplit = node.bestCovariate

        #if the data is continuous
        if self.dataTypeClassifier[nodeAttrSplit] == 1:
            
            #as there are only left and right branches possible with continuous branches
          
            #use casting to make a copy of the lists/dicts, otherwise it would create a pointer to the same memory location. 
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
        """
        method to get the node with the most people in it. 
        
        @params:
            node -> node in which we go through the branch lenghts.
        """
        keyWithMostPeople = ""
        length = 0
        
        #getting the branch with the biggest population
        for key in node.branches:
            temp = len(node.branches[key].populationAtNode)
            keyWithMostPeople = key if temp > length else keyWithMostPeople
        
        return keyWithMostPeople
    
    
    def measureGiniForCovariate(self, covariate , numberOfDivisions = 5):
        """
            This method is used to measure the gini index for a covariate. 
            
            it is divided into steps : 
                
                1. look at the type of covariate that we have on hand. 
                2. if continuous : 
                    2.0 -> create branch left and right in the node as there are only two ways possible
                    
                    2.1 -> split will take the form (x >= split) or (x < split)
                    
                    2.2 -> we need to find the best split within the possible continuous values. 
                    
                    2.3 -> in this implementation we proceed in a naive way by trying out a certain number of splits within
                    the continuous value. 
                    
                    2.4 -> store the gini given by each split and choose the smallest one after all computations. 
                3. if categorical : 
                    
                    3.0 -> create branches according to the posible values of the categorical covariate. 
                    
                    3.1 -> go through each member of the population at the node and place it in the corresponding 
                    branch. 
                    
                    3,2 -> measure the gini at each branch and add them up. 
                    
                4. return gini for this covariate. 
        """
        
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
            #dictionnary that will contain a nested dict of every member that has the same value for that covariate. 

            #structure -> dict[key] = {[index], value of attribute we are trying to predict.}
                # key = common value
                # index = index of sample
                # index is linked with value of attr we are trying to predict. 
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
    
    
    def getBranchWithMostElements(self, dictBranch):
        """
        Get the branch with the most element
        This method is only used for the measurement of the best possible split.
        
        @params:
            dictBranch -> dictionnary at banch
        """
        keyWithMostElements = ""
        numberOfElements = 0;
        
        for key in dictBranch:
            if len(dictBranch[key]) > numberOfElements:
                keyWithMostElements = key
            
        return keyWithMostElements
    


    def giniAtNode(self, population , numberOfPeopleAtNode):
        """
        this method is used to get the gini index at a specific Node. 
        
        We calculate the Gini using the formula ->
        Gini at Node = 1 - (probability of yes)^2 - (probability of no)^2
        
        It is a weighted index and so we multiply it with the length of the population
        sample at the node. 
        
        @params:
            population -> population dictionnary {index , value of attr we are trying to predict}
            used to get the gini index. 
            
            numberOfPeopleAtNode -> the length of the population
        """
        peopleYes = 0;
        for key in population:
            if population[key] != 0:
                peopleYes += 1
        
        peopleNo = numberOfPeopleAtNode - peopleYes
        probabilityYes = peopleYes/numberOfPeopleAtNode
        probabilityNo = peopleNo/numberOfPeopleAtNode
        
        gini = numberOfPeopleAtNode*(1 - (probabilityYes)**2 - probabilityNo**2)
        
        return gini
        
    
    def getBestPossibleSplitFromPossibleSplits(self, node, percentage = 0.8):
        """
        this method is used to tri within the gini indexes in the remaining splits to 
        find what which one would be best for the population
        
        
        reduced Remaining split quantity is set at 80% of the available remainingSplits
        can be changed
        
        @params:
            node -> node from which we want to get the best split
            
            percentage -> ammount of splits we want to keep from the remaining Splits
        """
        
        #list that will contain ginis
        giniWithCovariate = {}
        
        #picking at random from the availableSplits to add randomness. 
        newLength = int(percentage*len(node.remainingSplits))
        reducedRemainingSplits = list(random.sample(node.remainingSplits, newLength))
        
        for split in reducedRemainingSplits:
            giniWithCovariate[split] = self.measureGiniForCovariate(split)
            
            
        #simple variable to keep track of smallest gini
        covariate = ""
        smallestGini = -1
        g = 0
        
        
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
    
    






