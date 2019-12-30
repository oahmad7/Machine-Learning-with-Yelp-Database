# import the necessary packages
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from pyimagesearch import datasets
from pyimagesearch import models
import numpy as np
import argparse
import locale
import os
import time
import pandas as pd

# start a timer
startTimer =  time.time()

# load data
print("[INFO] loading restaurant attributes...")
df = pd.read_csv('NNBusinessedit.csv')

# construct a training and testing split with 75% of the data used
# for training and the remaining 25% for evaluation
print("[INFO] constructing training/testing split...")
(train, test) = train_test_split(df, test_size=0.25, random_state=42)

trainY = train["RestaurantsPriceRange2"]
testY = test["RestaurantsPriceRange2"]

# process the restaurant attributes data by performing min-max scaling
# on continuous features, one-hot encoding on categorical features,
# and then finally concatenating them together
print("[INFO] processing data...")
(trainX, testX) = datasets.process_restaurants_attributes(df, train, test)

# create our MLP and then compile the model using mean absolute
# percentage error as our loss, implying that we seek to minimize
# the absolute percentage difference between our price *predictions*
# and the *actual prices*
model = models.create_mlp(trainX.shape[1], regress=True)
opt = Adam(lr=1e-3, decay=1e-3 / 200)
model.compile(loss="mean_absolute_percentage_error", optimizer=opt)

# train the model
print("[INFO] training model...")
model.fit(trainX, trainY, validation_data=(testX, testY),
	epochs=200, batch_size=8)

# make predictions on the testing data
print("[INFO] predicting restaurant prices...")
preds = model.predict(testX)

# compute the difference between the *predicted* restaurant prices and the
# *actual* restaurant prices, then compute the percentage difference and
# the absolute percentage difference
diff = preds.flatten() - testY
percentDiff = (diff / testY) * 100
absPercentDiff = np.abs(percentDiff)

# compute the mean and standard deviation of the absolute percentage
# difference
mean = np.mean(absPercentDiff)
std = np.std(absPercentDiff)

# show some statistics on our model
locale.setlocale(locale.LC_ALL, "en_US.UTF-8")
print("[INFO] avg. restaurant price: {}, std restaurant price: {}".format(
	(df["RestaurantsPriceRange2"].mean()),
	(df["RestaurantsPriceRange2"].std())))
print(f"[INFO] mean: {mean}, std: {std}")

# end timer and print runtime
endTimer = time.time()
finalTime = (endTimer - startTimer)
print("[INFO] Runetime: " + str(finalTime) + " seconds \n")
