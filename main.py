import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model

# datasets descrip:
# 292 columns (291 attrs + 1 target)
# 30471 rows in train, 7662 in test
# test/(train+test): .20
# our target variable is price_doc
traindf = pd.read_csv('train.csv')
testdf = pd.read_csv('test.csv')

# print the shape of the datasets
print(traindf.shape)
print(testdf.shape)

# print all columns
print(traindf.columns)
