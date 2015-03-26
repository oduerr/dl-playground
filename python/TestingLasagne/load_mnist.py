import cPickle as pickle
import gzip
import numpy as np
import lasagne
import theano
import theano.tensor as T

import time

'''
    Using the mnist example with the network class
'''

def _load_data(file):
    with gzip.open(file, 'rb') as f:
        data = pickle.load(f)
    print ("Loaded data")
    return data

def load_data(file):
    data = _load_data(file)
    X,y = data[0]
    for i in range(1,len(data)): #We put all together since the splitting into training and test set is done anyway
        xc, yc = data[i]
        X = np.vstack((X, xc))
        y = np.hstack((y, yc))
    X = X.astype(np.float32)
    y = y.astype(np.int32)
    print ("Shape of X " +   str(X.shape))
    print ("  Min / Max X " + str(np.min(X)) + " " + str(np.max(X)))
    print ("Shape of Y " + str(y.shape))
    print ("  Min / Max Y " + str(np.min(y)) + " " + str(np.max(y)))
    return X,y

def load_data_2d(file):
    X, y = load_data(file) #
    PIXELS = int(np.sqrt(X.shape[1]))

    # Batch normalization
    Xmean = X.mean(axis = 0)
    XStd = np.sqrt(X.var(axis=0))
    X = (X-Xmean)/(XStd + 0.01)

    print ("After Batchnormalization Min / Max X / Mean " + str(np.min(X)) + " / " + str(np.max(X)) + " / " + str(np.mean(X)))
    X = X.reshape(-1, 1, PIXELS, PIXELS) #(70000, 1, 28, 28)
    return X, y, PIXELS



if __name__ == '__main__':
    X,y = load_data_2d(file='../../data/mnist.pkl.gz')
    print("Loaded data")
