import dill
from sklearn import base
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor

class ColumnSelectTransformer(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, col_names):
        self.col_names = col_names

    def fit(self, X, y=None):
        # This transformer doesn't need to learn anything about the data,
        # so it can just return self without any further processing
        return self

    def transform(self, X):
        self.X = X
        self.tx_undata = [] #not formatted so UNformatted DATA
        for i in self.col_names:
            self.tx_undata.append([row[i] for row in self.X])

        self.len_col = len(self.tx_undata)

        self.tx_data = []
        for i in self.tx_undata[0]:
            self.tx_data.append([i])
        for i in range(1, len(self.col_names)):
            for j in range(len(self.tx_data)):
                self.tx_data[j].append(self.tx_undata[i][j])

        # Return an array with the same number of rows as X and one
        # column for each in self.col_names
        return self.tx_data


class DictAttEncoder(base.BaseEstimator, base.TransformerMixin):

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        self.at_lt = X   # attribute list
        self.at_dt = []  # attribute flat dictionary
        for i in self.at_lt:
            for j in i:
                self.at_dt.append({})
                for k, l in j.items():
                    if not type(l) is dict:
                        if l == False:
                            self.at_dt[-1][k] = 0
                        elif l == True:
                            self.at_dt[-1][k] = 1
                        else:
                            self.at_dt[-1]['{}_{}'.format(k, l)] = 1
                    if type(l) is dict:
                        for m, n in l.items():
                            if n == False:
                                self.at_dt[-1]['{}_{}'.format(k, m)] = 0
                            elif n == True:
                                self.at_dt[-1]['{}_{}'.format(k, m)] = 1

        return self.at_dt

    # X will come in as a list of lists of lists.  Return a list of
    # dictionaries corresponding to those inner lists.



data = dill.load(open('data.pkd', 'rb'))

col_in = ['attributes']
cst = ColumnSelectTransformer(col_in)
cst2 = cst.fit_transform(data)

for i in range(0,5):
    print(cst2[i])
print(type(cst2))
#print(cst2[0:3])

cst3 = DictAttEncoder().fit_transform(cst2)
# cst3 = DictEncoder().fit_transform([[['a']], [['b', 'c']]])

for i in range(0,5):
    print(cst3[i])

# print(cst3)

col_in = ['attributes']
pipe3 = Pipeline([('column tx', ColumnSelectTransformer(col_in)),
                  ('dict fixer', DictAttEncoder()),
                  ('dict vector', DictVectorizer()),
                 ('forest model', RandomForestRegressor(n_estimators=100))])
