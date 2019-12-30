# import the necessary packages
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
import numpy as np
import glob
import cv2
import os

def load_restaurants_attributes(inputPath):
	# initialize the list of column names in the CSV file and then
	# load it using Pandas
	cols = ["RestaurantsPriceRange2", "GoodForKids", "city", "RestaurantsReservations", "HasTV", "state", "review_count", "RestaurantsTakeOut", "RestaurantsDelivery", "stars"]
	df = pd.read_csv(inputPath, sep=" ", header=None, names=cols)

	# return the data frame
	return df

def process_restaurants_attributes(df, train, test):
	# initialize the column names of the continuous data
	continuous = ["review_count", "stars"]
	categorical = ["RestaurantsPriceRange2", "city", "GoodForKids", "RestaurantsReservations", "HasTV", "state", "RestaurantsTakeOut", "RestaurantsDelivery"]

	# performin min-max scaling each continuous feature column to
	# the range [0, 1]
	cs = MinMaxScaler()
	trainContinuous = cs.fit_transform(train[continuous])
	testContinuous = cs.transform(test[continuous])

	# one-hot encode the categorical data (by definition of
	# one-hot encoing, all output features are now in the range [0, 1])
	binarizer1 = LabelBinarizer().fit(df["RestaurantsPriceRange2"])
	trainCategorical1 = binarizer1.transform(train["RestaurantsPriceRange2"])
	testCategorical1 = binarizer1.transform(test["RestaurantsPriceRange2"])

	binarizer2 = LabelBinarizer().fit(df["city"])
	trainCategorical2 = binarizer2.transform(train["city"])
	testCategorical2 = binarizer2.transform(test["city"])

	binarizer3 = LabelBinarizer().fit(df["GoodForKids"])
	trainCategorical3 = binarizer3.transform(train["GoodForKids"])
	testCategorical3 = binarizer3.transform(test["GoodForKids"])

	binarizer4 = LabelBinarizer().fit(df["RestaurantsReservations"])
	trainCategorical4 = binarizer4.transform(train["RestaurantsReservations"])
	testCategorical4 = binarizer4.transform(test["RestaurantsReservations"])

	binarizer5 = LabelBinarizer().fit(df["HasTV"])
	trainCategorical5 = binarizer5.transform(train["HasTV"])
	testCategorical5 = binarizer5.transform(test["HasTV"])

	binarizer6 = LabelBinarizer().fit(df["state"])
	trainCategorical6 = binarizer6.transform(train["state"])
	testCategorical6 = binarizer6.transform(test["state"])

	binarizer7 = LabelBinarizer().fit(df["RestaurantsTakeOut"])
	trainCategorical7 = binarizer7.transform(train["RestaurantsTakeOut"])
	testCategorical7 = binarizer7.transform(test["RestaurantsTakeOut"])

	binarizer8 = LabelBinarizer().fit(df["RestaurantsDelivery"])
	trainCategorical8 = binarizer8.transform(train["RestaurantsDelivery"])
	testCategorical8 = binarizer8.transform(test["RestaurantsDelivery"])

	# concatenate our data
	trainCategorical = np.hstack([trainCategorical1, trainCategorical2, trainCategorical3, trainCategorical4, trainCategorical5, trainCategorical6, trainCategorical7, trainCategorical8])
	testCategorical = np.hstack([testCategorical1, testCategorical2, testCategorical3, testCategorical4, testCategorical5, testCategorical6, testCategorical7, testCategorical8])

	# construct our training and testing data points by concatenating
	# the categorical features with the continuous features
	trainX = np.hstack([trainCategorical, trainContinuous])
	testX = np.hstack([testCategorical, testContinuous])

	# return the concatenated training and testing data
	return (trainX, testX)
