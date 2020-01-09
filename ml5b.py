from sklearn import base
from sklearn.ensemble import RandomForestRegressor


class DataAndResidualRegressor(base.BaseEstimator, base.TransformerMixin):

    #     def __init__(self):
    #         #its blank

    def fit(self, X, y):
        self.data = X
        self.y_act = y

        self.data_model1 = RandomForestRegressor(n_estimators=100)
        self.data_model2 = Ridge(alpha=1.0)
        # self.data_model2 = RandomForestRegressor(n_estimators=10)

        self.data_model1.fit(self.data, self.y_act)
        self.y_est = self.data_model1.predict(self.data)
        self.y_res = self.y_act - self.y_est

        self.data_model2.fit(self.data, self.y_res)
        self.y_res_est = self.data_model2.predict(self.data)

        return self

    def predict(self, X):
        self.y_predict = self.data_model1.predict(X) + self.data_model2.predict(X)

        return self.y_predict
