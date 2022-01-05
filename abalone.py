# -*- coding: utf-8 -*-
"""
Created on Tue Jan  4 19:26:42 2022

@author: Nabil Amimer
"""

import pandas as pd
import RandomForest as RF
import random
import time

start = time.time()

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
RF.defineDataTypeClassifier(dataFrame, dataTypeClassifier)

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
rf = RF.RandomForest(dataFrameTrain, dataTypeClassifier, 50, "rings")    
rf.predict(member)
end = time.time()

#print compilation time. 
print("compilation time : " + str(end - start)) 



