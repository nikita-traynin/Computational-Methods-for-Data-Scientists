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

# print(traindf.describe())

# # print the shape of the datasets
# print(traindf.shape)
# print(testdf.shape)
#
# # we have 157 ints, 16 non-scalars(strings), and 119 floats.
print(get_types(traindf)[0])
# print(get_types(traindf)[1])
# print(list(traindf.columns))

# convert the timestamp to an integer - representing unix seconds
# print(pd.to_datetime(traindf['timestamp']).values.astype(float)/(10**9))
traindf['timestamp'] = pd.to_datetime(traindf['timestamp']).values.astype(float)/(10**9)

# remove the id variable, we already have pandas row numbering
traindf = traindf.iloc[:, 1:]

# get the first 12 numeric types
first_10_vars = traindf.select_dtypes(include = 'number', exclude = 'O').iloc[:, 0:12]
# print(first_10_vars.head(10))

#unfortunately, it contains 'material', encoded as int though it is categorical. also remove timestamp
first_10_vars = first_10_vars.loc[:, (first_10_vars.columns != 'material') & (first_10_vars.columns != 'timestamp')]
# print(first_10_vars.head(10))

# plot histograms
fig, axs = plt.subplots(1, 5)
num_bins = 8
for i, column in enumerate(first_10_vars.columns[:5]):
    axs[i % 5].hist(first_10_vars[column], num_bins, facecolor='blue', alpha = 0.65, linewidth = 1, edgecolor = 'black')
    axs[i % 5].set_title(column)
plt.show()
fig, axs = plt.subplots(1, 5)
num_bins = 8
for i, column in enumerate(first_10_vars.columns[5:10]):
    axs[i % 5].hist(first_10_vars[column], num_bins, facecolor='blue', alpha = 0.65, linewidth = 1, edgecolor = 'black')
    axs[i % 5].set_title(column)
plt.show()
