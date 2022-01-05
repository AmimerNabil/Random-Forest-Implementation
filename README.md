# Random Forest Classifier Implementation 🌲

### Description ###
Hello to everyone! This project is a simple implementation of a Random forest from scratch using python. The goal of this project is to make it from scratch in order to understand the backdoor of this  machine learning tool. 

I got interested in this subject right after my first class of probability and statistics (fall 2021) and I am still fresh to the subject. However after viewing the 

### Random Forests ###
Random Forests are a machine learning tool used for both classification and regression problems. They are currently unexcelled in accuracy among current algorithms.  This forest is built using many classification tree which helps saves us from the task of pruning each tree. 

for more information and details on RF : [Random Forest Documentation](https://www.stat.berkeley.edu/~breiman/RandomForests/cc_home.htm)

##### details of implementation ->
The random forest implemented follows the CART protocol for random forest creation.  We first split the data into a boostrap dataset by randomly selecting _with replacement_ data from the dataset and then creating the branches of the tree by selecting the best split. 

The best split is determined using:
* for continuous : 
	* the minimum squared error
* for categorical :
	* the gini index

when creating the branches of the random forest, missing values are handled in a simplified way which can be found in the DecisionTree.py class under the "createBranches" method. 

There is indeed a lot of room for improvement (which I plan on doing) but the general purpose has been accomplished. 

### structure of the project ###

-> The DataSet Folder : it contains the data sets that used for testing the random forest. 

-> [RandomForest.py, DecisionTree.py, Nodes.py] : files used to create an instance of a random froest.

#### how  to use it : 
The random forest now works and can be used to create random forest classifiers. <hr> 
To use the random forest, 
0. import the necessary modules :
```python
# -*- coding: utf-8 -*-
import pandas as pd
import RandomForest as RF	
```

1.  create the Pandas DataFrame:
```python
#creation of the pandas dataFrame
dataFrame = pd.read_csv("DataSets/processed.cleveland.data")
```

2. create a datatypeClassifier :
```python
#a list which contains the type of attribute present in our dataframe.
'''
this list only contains whether a columns/covariate
contains categorical or continous variables
0 : categorical || 1 : continuous
'''

dataTypeClassifier = {}

RF.defineDataTypeClassifier(dataFrame, dataTypeClassifier)

```

Doing this step before creating the random forest makes it possible for you to make any changes to the type of variable in the dataTypeClassifier. 

This implementation has a specific way of determining whether a covariate is continuous or categorical (can be found in the Decision.py in DefineDataTypeClassifier)

_you can make any changes to the type in this manner ->_
```python
dataTypeClassifier["name of attribute (covariate)"] = 0
```

3. define a training and test set for the random forest 
```python
#splitting of the dataframe into a training and testing set.
dataFrameTrain = dataFrame.iloc[:4000].copy()
dataFrameTest = dataFrame.iloc[4000:].copy()
```

4. create the random forest using the training set. 
```python
rf = RF.RandomForest(dataFrameTrain, dataTypeClassifier, 50, "rings")

'''
you can then test it using some random index in your dataFrameTest set in this manner
'''
randomIndex = random.choice(dataFrameTest.index)
member = dataFrameTest.loc[randomIndex]

rf.predict(member)
```

### what is left to do?

* The current implementation only supports categorical random forests. However I would like to implement its counterpart for regression problems. 
* implement concrete evaluation tools for the random forest. 
* implement visual tools to help decoding Single Decision trees. 


### Liscence  : 
MIT License

Copyright (c) [2022] [Nabil Amimer]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:


