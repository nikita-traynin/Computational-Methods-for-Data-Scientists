import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model


# get the type of each column, as well as the distribution of types in the dataset
def get_types(df):
    # show the distribution of datatypes
    typesaggr = {}
    types = {}
    for i in df.columns:
        # count the aggregate types
        type = df.dtypes[i]
        if type not in typesaggr.keys():
            typesaggr[type] = 1
        else:
            typesaggr[type] += 1

        # pair type with variable name
        types[i] = type

    return typesaggr, types


# datasets descrip:
# 292 columns (291 attrs + 1 target)
# 30471 rows in train, 7662 in test
# test/(train+test): .20
# our target variable is price_doc
traindf = pd.read_csv('train.csv')
testdf = pd.read_csv('test.csv')
pd.set_option('display.max_columns', 300)
print(traindf.describe())

# # print the shape of the datasets
# print(traindf.shape)
# print(testdf.shape)
#
#
#
# # we have 157 ints, 16 non-scalars(strings), and 119 floats.
# print(get_types(traindf)[0])
# print(list(traindf.columns))

# encode the strings into categoricals, except for dates which we convert to ints
