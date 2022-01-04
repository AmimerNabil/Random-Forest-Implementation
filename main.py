# -*- coding: utf-8 -*-
"""
Created on Wed Dec 29 13:11:34 2021

@author: Nabil Amimer
"""
import pandas as pd
from RandomForest import RandomForest
import random

dataFrame = pd.read_csv("processed.cleveland.data")
dataFrame["num"] = dataFrame["num"].replace([2 , 3 , 4] , 1)

indexes = dataFrame.index
randomIndex = random.choice(indexes)


rf = RandomForest(dataFrame, 50, "num")    
print(randomIndex)
print(dataFrame.loc[randomIndex])
print(rf.OOBprediction(randomIndex))

