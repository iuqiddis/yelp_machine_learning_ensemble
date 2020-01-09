#import ujson as json
#import gzip
import dill
import pandas as pd

#with gzip.open('yelp_train_academic_dataset_business.json.gz') as f:
#    data = [json.loads(line) for line in f]

#dill.dump(data, open('data.pkd', 'wb'))
data = dill.load(open('data.pkd', 'rb'))

for i in data[17:20]:
    print(i)

#The line below is part of the instructions for cleaner represenation. i ignored it
star_ratings = [row['stars'] for row in data]

#################
# I did it this way for part 1, but part 2 generalizes it with star_ratings.
# Things I should have done correctly the first time around. Oh well.

# cysr = [] #city stars
# for dt in data: #dt is dictionary
#     for key in dt.keys():
#         if key == 'city':
#             cysr.append([dt['city']])
#         if key == 'stars':
#             cysr[-1].append(dt['stars'])
#
#
#
# cy, sr = [], []
# for i in cysr:
#     cy.append(i[0])
#     sr.append(i[1])
# ###################

cy, sr = [], [] #city stars

for dt in data: #dt is dictionary
    for key in dt.keys():
        if key == 'city':
            cy.append(dt['city'])
for i in star_ratings:
    sr.append(i)

spd = pd.DataFrame({ 'city': cy,
                     'stars': sr})

sad = spd.groupby(spd['city'], as_index=False).mean()

ca = sad['city'].values.tolist() #aggregate city
sa = sad['stars'].values.tolist()

avg_stars = {}
for i, j in zip(ca,sa):
        avg_stars[i] = j



#This is for question 1 formatting
# cst = len(ca)
# avg_stars = []
# for i in range(cst):
#     avg_stars.append((ca[i],sa[i]))


#print(avg_stars)