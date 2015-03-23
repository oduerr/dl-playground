"""
    Creates pickeld data set from lists of file, label
"""
import os

import numpy as np
from pandas.io.parsers import read_csv
import cv2
import pickle

SHOW = False;
laptop = True
pixels = 56


def createData(file, pixels = 56, laptop=False):
    df = read_csv(file, index_col=False, header=0, sep='\t')
    imageSize = pixels * pixels
    num_rows = df.shape[0]
    num_features = imageSize
    X = np.zeros((num_rows, num_features), dtype=float)
    y = np.zeros((num_rows))
    for row in df.iterrows():
        i = row[0]
        fn = row[1][0]
        # On the Laptop, replace part
        if laptop:
            fn = fn.replace('/home/dueo/data/inselpng/', '/Users/oli/Proj_Large_Data/Deep_Learning_MRI/inselpng/')
        label = int(row[1][1])
        img = cv2.imread(fn, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        img = cv2.resize(img, (pixels, pixels))
        X[i, 0:imageSize] = np.reshape(img, (1, imageSize))
        y[i] = label
        if (SHOW):
            cv2.imshow('Test', img)
            cv2.waitKey(10000)
    return (X,y)


(XTrain, yTrain) = createData(file='../MRI/training.txt', pixels =pixels, laptop=laptop)
print("Read the training file X = " + str(XTrain.shape) + " y=" + str(yTrain.shape))
(XTest, yTest) = createData(file='../MRI/testing.txt', pixels=pixels, laptop=laptop)
print("Read the test file X = " + str(XTest.shape) + " y=" + str(yTest.shape))
data = [[XTrain,yTrain],[XTest,yTest]]
with open('data/data' + str(pixels) + '.pkl', "wb") as f:
    pickle.dump(data, f)