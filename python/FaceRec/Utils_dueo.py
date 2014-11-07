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

def preprocess(vals, zca, sizeOut = 46, show = True):
        if (zca != None):
            X_white = zca.transform(vals)
        else:
            X_white = vals
        sizeIn = int(math.sqrt(len(vals)))
        img = np.reshape(vals, (sizeIn, sizeIn)) / 255.0
        mini = np.percentile(X_white, 0.01)
        maxi = np.percentile(X_white, 99.99)
        # X_white_rx = (X_white - mini) / (maxi - mini)  # Rescaling to be in between 0 and 1
        # X_white_i = np.array(X_white_rx * 255, dtype=np.uint8)
        # X_white1 = np.reshape(X_white_i, (sizeIn, sizeIn))
        #
        # img_small = cv2.resize(X_white1, (sizeOut + 2, sizeOut + 2))
        # # img_small = img #No resizing
        # img_small = cv2.equalizeHist(img_small)

        # model = cv2.createLBPHFaceRecognizer()
        # model.train([np.asarray(img_small)], np.asarray([42]))
        # dumm = model.getMatVector('histograms')[0]
        X = np.asarray(img)
        X = (1 << 7) * (X[0:-2, 0:-2] >= X[1:-1, 1:-1]) \
            + (1 << 6) * (X[0:-2, 1:-1] >= X[1:-1, 1:-1]) \
            + (1 << 5) * (X[0:-2, 2:] >= X[1:-1, 1:-1]) \
            + (1 << 4) * (X[1:-1, 2:] >= X[1:-1, 1:-1]) \
            + (1 << 3) * (X[2:, 2:] >= X[1:-1, 1:-1]) \
            + (1 << 2) * (X[2:, 1:-1] >= X[1:-1, 1:-1]) \
            + (1 << 1) * (X[2:, :-2] >= X[1:-1, 1:-1]) \
            + (1 << 0) * (X[1:-1, :-2] >= X[1:-1, 1:-1])


        # img_small = 255 * cv2.resize(dumm, (sizeOut, sizeOut))

        if (show):
            cv2.imshow('Original', cv2.resize(img, (280, 280)))
            cv2.imshow('LocalBinaryHists', cv2.resize(X / 255., (280, 280)))
            cv2.waitKey(1)
        return np.asarray(np.reshape(X, sizeOut ** 2), np.int)



# Loads the pictures and creates the data as needed for theano
def load_pictures():
    import sys
    filenameTesting    = "../../data/testing_48x48_unaligned_large.p_R.csv.gz"
    # We use the manipulated ones for training
    filenameValidation   = "../../data/training_48x48_aligned_large.p_R.csv.gz"
    filenameTraining = "../../data/training_48x48_aligned_large_expanded.p_R.csv.gz"

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
            cv2.namedWindow('LocalBinaryHists', cv2.WINDOW_NORMAL)
        minV = 1e100
        maxV = -1e100
        with gzip.open(filename) as f:
            reader = csv.reader(f)
            for row in reader:
                y_tmp.append(int(row[0]))
                vals = np.asarray(row[1:], np.int)
                preprocessed = preprocess(vals, zca)
                minV = min(min, np.amin(preprocessed))
                maxV = max(max, np.amax(preprocessed))
                x_tmp.append(preprocessed / 255.)
        print("  Data Range" + str(minV) + "  " + str(maxV))
        return (np.asarray(x_tmp, theano.config.floatX), np.asarray(y_tmp, theano.config.floatX))

    #zca = learnWhitening(filenameTraining)
    zca = None
    #test_set_all = loadFromCSV(filenameTesting, zca)
    #N = len(test_set_all[1])
    #valid = int(N * 0.2)

    print theano.config

    test_set = loadFromCSV(filenameTesting, zca)
    print(" Number of test examples [" + str(test_set[1].shape[0]) + "]")
    valid_set = loadFromCSV(filenameValidation, zca)
    print(" Number of validation examples [" + str(valid_set[1].shape[0]) + "]")
    import os.path, time
    print "Training set last modified: %s" % time.ctime(os.path.getmtime(filenameTraining))
    train_set = loadFromCSV(filenameTraining, zca)
    print(" Number of training examples [" + str(train_set[1].shape[0]) + "]")

    if (show):
        cv2.destroyAllWindows()

    #train_set, valid_set, test_set format: tuple(input, target)
    #input is an numpy.ndarray of 2 dimensions (a matrix)
    #witch row's correspond to an example. target is a
    #numpy.ndarray of 1 dimensions (vector)) that have the same length as
    #the number of rows in the input. It should give the target
    #target to the example with the same index in the input.

    def shared_dataset(data_xy, perm, borrow=True):
        """ Function that loads the dataset into shared variables

        The reason we store our dataset in shared variables is to allow
        Theano to copy it into the GPU memory (when code is run on GPU).
        Since copying data into the GPU is slow, copying a minibatch everytime
        is needed (the default behaviour if the data is not in a shared
        variable) would lead to a large decrease in performance.
        """
        data_x, data_y = data_xy
        shared_x = theano.shared(np.asarray(data_x[perm],
                                            dtype=theano.config.floatX),
                                 borrow=borrow)
        shared_y = theano.shared(np.asarray(data_y[perm],
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

    test_set_x, test_set_y = shared_dataset(test_set, np.random.permutation(test_set[1].shape[0]))
    valid_set_x, valid_set_y = shared_dataset(valid_set, np.random.permutation(valid_set[1].shape[0]))
    train_set_x, train_set_y = shared_dataset(train_set, np.random.permutation(train_set[1].shape[0]))

    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),
            (test_set_x, test_set_y)]
    return rval

if __name__ == "__main__":
    load_pictures()
