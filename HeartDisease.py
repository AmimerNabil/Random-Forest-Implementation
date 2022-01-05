# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:11:34 2021

@author: Nabil Amimer
"""
import pandas as pd
import RandomForest as RF
import random
import time

start = time.time()


#creation of the pandas dataFrame
dataFrame = pd.read_csv("processed.cleveland.data")

#in this context, it is useless for us to have the values 2,3,4 for the "num"
#attributes. replace all these values by 1.
dataFrame["num"] = dataFrame["num"].replace([2 , 3 , 4] , 1)

#a list which contains the type of attribute present in our dataframe. 
'''
    this list only contains whether a columns/covariate
    contains categorical or continous variables
    0 : categorical || 1 : continuous
'''
dataTypeClassifier = {}

RF.defineDataTypeClassifier(dataFrame, dataTypeClassifier)

#splitting of the dataframe into a training and testing set. 
dataFrameTrain = dataFrame.iloc[:250].copy()
dataFrameTest = dataFrame.iloc[250:].copy()


#to test the random froest, we can pick a random index from the test set and 
#run the "prediction()" method on it to retrieve "num". 
randomIndex = random.choice(dataFrameTest.index)
member = dataFrameTest.loc[randomIndex]


#creation of the random forest with 50 trees only. 
rf = RF.RandomForest(dataFrameTrain, dataTypeClassifier, 1, "num")    
prediction = rf.predict(member)

end = time.time()

#print compilation time. 
print("compilation time : " + str(end - start)) 



