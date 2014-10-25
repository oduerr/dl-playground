__author__ = 'oli'
import pickle

import numpy as np
import theano
import theano.tensor as T

import math
import csv
import gzip
import cv2

import ZCA

show = True

def load_pictures():
    import sys
    filenameTesting  = "../../data/testing_48x48_aligned_large.p_R.csv.gz"
    filenameTraining = "../../data/training_48x48_aligned_large.p_R.csv.gz"
    #filenameTraining = "../../data/training_48x48_aligned_large_expanded.p_R.csv.gz"

    def learnWhitening(filename):
        x_tmp = []
        with gzip.open(filename) as f:
            reader = csv.reader(f)
            for row in reader:
                vals = np.asarray(row[1:], np.int)
                x_tmp.append(np.asarray(row[1:], np.int))
        print("Read Image for Whitening transformation")
        ret = ZCA.ZCA()
        return (ret.fit(x_tmp))

    def loadFromCSV(filename, zca=None):
        y_tmp = []
        x_tmp = []
        if (show):
            cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Rescaled', cv2.WINDOW_NORMAL)
        with gzip.open(filename) as f:
            reader = csv.reader(f)
            for row in reader:
                y_tmp.append(int(row[0]))
                vals = np.asarray(row[1:], np.int)
                if (zca != None):
                    X_white = zca.transform(vals)
                else:
                    X_white = vals
                NDumm = int(math.sqrt(len(vals)))
                img = np.reshape(vals, (NDumm, NDumm)) / 255.0
                mini = np.percentile(X_white,  0.01)
                maxi = np.percentile(X_white,  99.99)
                X_white_rx = (X_white - mini) / (maxi - mini) #Rescaling to be in between 0 and 1
                X_white_i = np.array(X_white_rx * 255, dtype = np.uint8)
                X_white1 = np.reshape(X_white_i, (48,48))
                img_small = cv2.resize(X_white1, (28, 28))
                img_small = cv2.equalizeHist(img_small)
                #img_small = img #No resizing
                if (show):
                    cv2.imshow('Original', img)
                    cv2.imshow('Rescaled', cv2.resize(img_small, (280, 280)))
                    cv2.waitKey(1)
                vals = np.asarray(255 * np.reshape(img_small, 28 ** 2), np.int)
                print(str(np.amin(vals)) + "  " + str(np.amax(vals)))
                x_tmp.append(vals)
        return (np.asarray(x_tmp, theano.config.floatX), np.asarray(y_tmp, theano.config.floatX))

    zca = learnWhitening(filenameTraining)
    #zca = None
    test_set_all = loadFromCSV(filenameTesting, zca)
    N = len(test_set_all[1])
    valid = int(N * 0.2)
    print theano.config
    print(" Number of test examples [" + str(N) + "] loaded from " + filenameTesting + " using " + str(valid)  + " for validation. Type " + str(type(test_set_all)))

    perm = np.random.permutation(N)
    perm_valid = perm[0:valid]
    perm_rest  = perm[(valid+1):]
    #TODO OLIVER check if permutation is bug free
    valid_set = (np.take(test_set_all[0], perm_valid, 0), np.take(test_set_all[1], perm_valid, 0))
    test_set = (np.take(test_set_all[0], perm_rest,0), np.take(test_set_all[1],perm_rest, 0))
    train_set = loadFromCSV(filenameTraining, zca)
    if (show):
        cv2.destroyAllWindows()

    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y,
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

    test_set_x, test_set_y = shared_dataset(test_set)
    valid_set_x, valid_set_y = shared_dataset(valid_set)
    train_set_x, train_set_y = shared_dataset(train_set)

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

#load_pictures()
