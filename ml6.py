from sklearn import base
import pandas as pd
import dill
from sklearn.pipeline import FeatureUnion

class CityEstimator(base.BaseEstimator, base.RegressorMixin):

    def __init__(self):
        self.avg_stars = {}

    def fit(self, X, y):

        self.cy, self.sr = [], []  # city stars

        for dt in X:  # dt is dictionary
            for key in dt.keys():
                if key == 'city':
                    self.cy.append(dt['city'])
        for i in y:
            self.sr.append(i)

        self.spd = pd.DataFrame({'city': self.cy,
                                 'stars': self.sr})

        self.sad = self.spd.groupby(self.spd['city'], as_index=False).mean()

        self.ca = self.sad['city'].values.tolist()  # aggregate city
        self.sa = self.sad['stars'].values.tolist()

        # self.avg_stars = {}
        for i, j in zip(self.ca, self.sa):
            self.avg_stars[i] = j

        return self

    def predict(self, X):

        for i in X:  # the key is in a list
            for key in i.keys():
                if key == 'city':
                    if not i[key] in self.avg_stars.keys():
                        self.avg_stars[i[key]] = 2.5
                        # in X the city name is a value. But its a key for self.avg_stars
        return [self.avg_stars[row['city']] for row in X]



class EstimatorTransformer(base.BaseEstimator, base.TransformerMixin):

    def __init__(self, estimator):
        self.estimator = estimator
        #return self

    # What needs to be done here?

    def fit(self, X, y):
        self.estimator.fit(X,y)
        return self

    # Fit the stored estimator.
    # Question: what should be returned?

    def transform(self, X):
        self.y_est = self.estimator.predict(X)
        self.y_out = [[i] for i in self.y_est]

        return self.y_out

# Use predict on the stored estimator as a "transformation".
# Be sure to return a 2-D array.

############################################################

data = dill.load(open('data.pkd', 'rb'))
star_ratings = [row['stars'] for row in data]

city_est = CityEstimator()
city_trans = EstimatorTransformer(city_est)

city_trans.fit(data, star_ratings)
assert ([r[0] for r in city_trans.transform(data[:5])]
        == city_est.predict(data[:5]))



union = FeatureUnion([('city est', city_trans),
                      ('lat long', latlong_trans),
                      ('cat mod', cat_trans),
                      ('att mod', att_trans)])

