# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:11:34 2021

@author: Nabil Amimer
"""
import pandas as pd
from RandomForest import RandomForest
import random
import time

start = time.time()


#creation of the pandas dataFrame
dataFrame = pd.read_csv("processed.cleveland.data")

#in this context, it is useless for us to have the values 2,3,4 for the "num"
#attributes. replace all these values by 1.
dataFrame["num"] = dataFrame["num"].replace([2 , 3 , 4] , 1)


#splitting of the dataframe into a training and testing set. 
dataFrameTrain = dataFrame.iloc[:250].copy()
dataFrameTest = dataFrame.iloc[250:].copy()


#to test the random froest, we can pick a random index from the test set and 
#run the "prediction()" method on it to retrieve "num". 
randomIndex = random.choice(dataFrameTest.index)
member = dataFrameTest.loc[randomIndex]

#creation of the random forest with 50 trees only. 
rf = RandomForest(dataFrameTrain, 100, "num")    

#print the test results
print(randomIndex)
print(dataFrame.loc[randomIndex])
print(rf.predict(member))

end = time.time()

#print compilation time. 
print("compilation time : " + str(end - start)) 



