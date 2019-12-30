#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
import json
import random
import re
from collections import defaultdict, Counter
from functools import reduce

# from IPython.display import HTML, display
import numpy as np
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import cross_val_score
import tabulate


objs = []

with open("./business.json") as f:
    # Each line in business.json is a separate json object
    for line in f:
        # Read the line, convert it into a dictionary with json.loads,
        # then convert THAT into a defaultdict so that if we ask for a
        # field that a line doesn't have, it will give an empty string
        # instead of throwing an error
        objs.append(defaultdict(lambda: "", json.loads(line)))


# Get all unique columns
keys = set()
for obj in objs:
    keys.update(obj.keys())


# Split categories into list
for obj in objs:
    # This would be easier, but no guarantee categories always split by ", "
    # and not just ","
    # obj["categories"] = obj["categories"].split(", ")
    cats = obj["categories"]
    # Might be null
    if cats:
        obj["categories"] = [s.strip() for s in cats.split(",")]
    else:
        obj["categories"] = []


# Filter just restaurants
restaurants = [o for o in objs if "Restaurants" in o["categories"]]


# Promote "attributes" fields to full-blown columns
for r in restaurants:
    attrs = r["attributes"]
    if not attrs:
        continue
    for k in attrs.keys():
        keys.add(k)
        r[k] = attrs[k]
keys.remove("attributes") # Don't want in final table


# Filter just restaurants with prices
restaurants = [r for r in restaurants if r["RestaurantsPriceRange2"] in ["1", "2", "3", "4"]]


# END DATA CLEANUP
# START TIMER

start=datetime.now()


prices = [r["RestaurantsPriceRange2"] for r in restaurants]


valRE = re.compile("u?'(.+)'")
def cleanVal(val):
    m = valRE.match(val)
    if m:
        return m[1]
    return val


def columnEncode(colName):
    col = [cleanVal(r[colName]) for r in restaurants]
    binCol = LabelBinarizer().fit_transform(col)
    return np.array(binCol)


multinomialColumnNames = [
    'AcceptsInsurance',
    'AgesAllowed',
    'Alcohol',
    'BYOB',
    'BYOBCorkage',
    'BikeParking',
    'BusinessAcceptsBitcoin',
    'BusinessAcceptsCreditCards',
    'ByAppointmentOnly',
    'Caters',
    'CoatCheck',
    'Corkage',
    'DogsAllowed',
    'DriveThru',
    'GoodForDancing',
    'GoodForKids',
    'HappyHour',
    'HasTV',
    'NoiseLevel',
    'Open24Hours',
    'OutdoorSeating',
    'RestaurantsAttire',
    'RestaurantsCounterService',
    'RestaurantsDelivery',
    'RestaurantsGoodForGroups',
    'RestaurantsReservations',
    'RestaurantsTableService',
    'RestaurantsTakeOut',
    'Smoking',
    'WheelchairAccessible',
    'WiFi'
]


rowOrder = list(range(len(prices)))
random.shuffle(rowOrder)
breakPoint = int(0.75*len(prices))
trainIndices = rowOrder[:breakPoint]
testIndices = rowOrder[breakPoint:]

trainPrices = np.array(prices)[trainIndices]


multinomialEncodedCols = {}
for c in multinomialColumnNames:
    multinomialEncodedCols[c] = columnEncode(c)


# Taken individually, features are not very meaningful
# Skip cross-validation when not generating table - slow
# table = [[c,
#           np.mean(cross_val_score(MultinomialNB(), multinomialEncodedCols[c], prices, cv=10, n_jobs=-1)),
#           np.mean(cross_val_score(ComplementNB(), multinomialEncodedCols[c], prices, cv=10, n_jobs=-1))]
#         for c in multinomialColumnNames]
# display(HTML(tabulate.tabulate(table, tablefmt='html')))


multinomialModels = {}
for c in multinomialColumnNames:
    col = np.array(multinomialEncodedCols[c])[trainIndices]
    multinomialModels[c] = MultinomialNB()
    multinomialModels[c].fit(col, trainPrices)


# Replaced by numpy
# def metaPredict(i):
#     prob = np.array([1,1,1,1])
#     for colName in multinomialColumnNames:
#         val = np.array(multinomialEncodedCols[colName][i]).reshape(1,-1)
#         model = multinomialModels[colName]
#         newProbs = model.predict_proba(val)
#         prob = np.multiply(prob, newProbs)
#     return prob


priceOpts = ["1", "2", "3", "4"]


probList = [multinomialModels[c].predict_proba(multinomialEncodedCols[c][testIndices]) for c in multinomialColumnNames]
predictedProbs = reduce(np.multiply, probList)

getCategory = lambda vec: priceOpts[np.argmax(vec)]
predictions = np.apply_along_axis(getCategory, axis=1, arr=predictedProbs)


right = 0
wrong = 0
for (p, a) in zip(predictions, np.array(prices)[testIndices]):
    if p == a:
        right += 1
    else:
        wrong += 1


print("Multinomial model accuracy:")
print(wrong / (right + wrong))

print("Time")
print(datetime.now() - start)
print("s")

start = datetime.now()


complementModels = {}
for c in multinomialColumnNames:
    col = np.array(multinomialEncodedCols[c])[trainIndices]
    complementModels[c] = ComplementNB()
    complementModels[c].fit(col, trainPrices)


def metaPredictComplement(i):
    prob = np.array([1,1,1,1])
    for colName in multinomialColumnNames:
        val = np.array(multinomialEncodedCols[colName][i]).reshape(1,-1)
        model = complementModels[colName]
        newProbs = model.predict_proba(val)
        prob = np.multiply(prob, newProbs)
    return prob


rightC = 0
wrongC = 0
for i in testIndices:
    truePrice = prices[i]
    predPrice = priceOpts[np.argmax(metaPredictComplement(i))]
    if truePrice == predPrice:
        rightC += 1
    else:
        wrongC += 1


print("Complement model accuracy:")
print(wrongC / (rightC + wrongC))

print("Time")
print(datetime.now() - start)
print("s")
# In[201]:


(wrongC) / (rightC + wrongC) * 100


# In[221]:


start = datetime.now()


nnColNames = [
    'city',
    'GoodForKids',
    'RestaurantsReservations',
    'HasTV',
    'state',
    'review_count',
    'RestaurantsTakeOut',
    'RestaurantsDelivery',
    'stars'
]

def isNNRestaurant(r):
    for c in nnColNames:
        if c not in r or r[c] == '' or r[c] == 'None':
            return False
    return True

nnRestaurants = [r for r in restaurants if isNNRestaurant(r)]
len(nnRestaurants)


# In[241]:


nnPrices = [r["RestaurantsPriceRange2"] for r in nnRestaurants]


# In[235]:


def toBin(str):
    s = str.strip().lower()
    if s == 'true':
        return 1
    elif s == 'false':
        return 0
    else:
        raise Exception(s)

nnFVector = [[toBin(r["GoodForKids"]),
              toBin(r["RestaurantsReservations"]),
              toBin(r["HasTV"]),
              toBin(r["RestaurantsTakeOut"]),
              toBin(r["RestaurantsDelivery"])] for r in nnRestaurants]


# In[239]:


nnState = LabelBinarizer().fit_transform([r["state"].upper() for r in nnRestaurants])


# In[240]:


nnGVector = [[r["review_count"], r["stars"]] for r in nnRestaurants]


# In[243]:


nnRowOrder = list(range(len(nnPrices)))
random.shuffle(nnRowOrder)
nnBreakPoint = int(0.75*len(nnPrices))

nnTrainIndices = nnRowOrder[:nnBreakPoint]
nnTestIndices = nnRowOrder[nnBreakPoint:]

nnTrainPrices = np.array(nnPrices)[nnTrainIndices]


# In[246]:


binNB = BernoulliNB()
binNB.fit(np.array(nnFVector)[nnTrainIndices], nnTrainPrices)


# In[247]:


stateNB = MultinomialNB()
stateNB.fit(np.array(nnState)[nnTrainIndices], nnTrainPrices)


# In[248]:


numNB = GaussianNB()
numNB.fit(np.array(nnGVector)[nnTrainIndices], nnTrainPrices)


# In[255]:


Counter(binNB.predict(np.array(nnFVector)[nnTestIndices]))


# In[256]:


Counter(stateNB.predict(np.array(nnState)[nnTestIndices]))


# In[257]:


Counter(numNB.predict(np.array(nnGVector)[nnTestIndices]))


# In[259]:


binaryPred = binNB.predict_proba(np.array(nnFVector)[nnTestIndices])
statePred = stateNB.predict_proba(np.array(nnState)[nnTestIndices])
numericPred = numNB.predict_proba(np.array(nnGVector)[nnTestIndices])

testMetaPredictions = np.multiply(binaryPred, statePred, numericPred)


# In[262]:


getCategory = lambda vec: priceOpts[np.argmax(vec)]
nnPredictions = np.apply_along_axis(getCategory, axis=1, arr=testMetaPredictions)


# In[277]:


right = 0
wrong = 0
for (p,a) in zip(nnPredictions, np.array(nnPrices)[nnTestIndices]):
    if p == a:
        right += 1
    else:
        wrong += 1


# In[278]:

print("Dense accuracy:")
print(wrong / (right + wrong) * 100)

print("Time")
print(datetime.now() - start)
print("s")

