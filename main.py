import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import linear_model
from sklearn import preprocessing
from sklearn import metrics
import seaborn as sn

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

# fix broken years
traindf.at[15220, 'build_year'] = 1965
traindf.at[10089, 'build_year'] = 2009
pd.set_option('display.max_columns', 15)

# print(traindf.describe())

# # print the shape of the datasets
# print(traindf.shape)
# print(testdf.shape)
#
# # we have 157 ints (includes id), 16 non-scalars(strings, includes timestamp), and 119 floats.
# print(get_types(traindf)[0])
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

# # plot histograms
# fig, axs = plt.subplots(1, 5)
# num_bins = 8
# for i, column in enumerate(first_10_vars.columns[:5]):
#     axs[i % 5].hist(first_10_vars[column], num_bins, facecolor='blue', alpha = 0.65, linewidth = 1, edgecolor = 'black')
#     axs[i % 5].set_title(column)
# plt.show()
# fig, axs = plt.subplots(1, 5)
# num_bins = 8
# for i, column in enumerate(first_10_vars.columns[5:10]):
#     axs[i % 5].hist(first_10_vars[column], num_bins, facecolor='blue', alpha = 0.65, linewidth = 1, edgecolor = 'black')
#     axs[i % 5].set_title(column)
# plt.show()

# encode booleans
cats = traindf.select_dtypes(include='O')
catdict = {}
for column in cats.columns:
    num_cats = len(set(cats[column]))
    catdict[column] = num_cats
    # if its a binary variable, simply enumerate it as 0s and 1s
    if num_cats == 2:
        traindf[column] = pd.factorize(traindf[column])[0]

# print(catdict)

# count number of classes for categoricals (that aren't binary)
cats = traindf.select_dtypes(include='O')
# print(cats.head(15))

# there's 146 cats, so drop 'sub_area'
traindf = traindf.loc[:, traindf.columns != 'sub_area']

# now, we only have one string-encoded categorical. OneHotEncoder
one_hot = pd.get_dummies(traindf['ecology'])
traindf = traindf.drop('ecology', axis=1)
traindf = traindf.join(one_hot)

# ratio of numerics to all columns (should be one)
# print('\n\n\n' + str(len(traindf.select_dtypes('number').columns) / len(traindf.columns)))

# we know should have extra columns with our one hot variable
# print(len(traindf.columns))
# print(sorted(list(traindf['build_year'].dropna()))[-10:])

# now we impute all nans.
traindf = traindf.fillna(traindf.median(axis=0, skipna=True))
# print(list(traindf.mean(axis=0, skipna=True)))


# print(traindf.iloc[:,1:15].head(15))

# scale!
scaler = preprocessing.StandardScaler()
scaler.fit(traindf)
traindf_scaled = scaler.transform(traindf)

# print(traindf.head(25))

# print(traindf_scaled.mean(axis=0))
# print(traindf_scaled.std(axis=0))


# only keep correlated variables
corrs = []
high_corrs = {}
traindf_scaled_small = np.zeros((30471, 1))
for i in range(traindf_scaled.shape[1]-1):
    corrs.append(np.corrcoef(traindf_scaled[:, i], traindf_scaled[:, -1])[1, 0])
    if corrs[i] >= 0.15:
        traindf_scaled_small = np.concatenate((traindf_scaled_small, np.expand_dims(traindf_scaled[:, i], axis=1)), axis=1)
        high_corrs[i] = corrs[i]

# create dataset with only important variables
traindf_scaled_small = np.delete(traindf_scaled_small, 0, 1)

# print the highest correlations
print(high_corrs)

#make the correlation heat map
cols = traindf.columns[[(i in high_corrs.keys()) for i in range(traindf_scaled.shape[1])]]
corrMatrix = pd.DataFrame(traindf_scaled_small).set_axis(cols, axis=1, inplace=False).corr()
sn.heatmap(corrMatrix, annot=True)
plt.show()


# NOW finally we can apply the model
lr = linear_model.SGDRegressor(loss='squared_loss', alpha=0, max_iter=20000, verbose=10, learning_rate='constant', eta0=.001, early_stoppin=True)
X = traindf_scaled_small[:, :-1]
y = traindf_scaled_small[:, -1]

# double check the shape of our final training set
print(np.shape(X), np.shape(y))

# fit the model to this training set
lr.fit(X=X, y=y)

# get TRAIN predictions
y_pred = lr.predict(X)

# make sure our predictions look right
# print(y_pred.shape)

# get the r2 score
print(type(y))
print(type(y_pred))
print('r^2 score:' + str(lr.score(X, y)))

# we cannot do f1 score because this is a continuous target variable
