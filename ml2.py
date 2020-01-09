from sklearn import base
import pandas as pd
import dill

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



data = dill.load(open('data.pkd', 'rb'))
star_ratings = [row['stars'] for row in data]

# Initializing Model
city_est = CityEstimator()
city_est.fit(data, star_ratings)

# Testing with original data
city_out = city_est.predict(data[:5])
print(city_out)

city_out2 = [[i] for i in city_out]
print(city_out2)
# Testing with a new case
#print(city_est.predict([{'city': 'Timbuktu'}]))
