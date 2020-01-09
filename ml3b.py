import dill
from sklearn import base
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsRegressor

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


data = dill.load(open('data.pkd', 'rb'))

col_in = ['latitude', 'longitude']

pipe = Pipeline([('column tx', ColumnSelectTransformer(col_in)),
                 ('k near', KNeighborsRegressor(n_neighbors=5))])

