__author__ = 'oli'

import math
import csv
import gzip

import numpy as np
import theano
import theano.tensor as T
import cv2

show = False
filenameTesting    = "/Users/oli/tmp/kaggle_plankton/train/"
filenameValidation = "/Users/oli/tmp/kaggle_plankton/test/"


def LBH_Norm(X):
    X = (1 << 7) * (X[0:-2, 0:-2] >= X[1:-1, 1:-1]) \
        + (1 << 6) * (X[0:-2, 1:-1] >= X[1:-1, 1:-1]) \
        + (1 << 5) * (X[0:-2, 2:] >= X[1:-1, 1:-1]) \
        + (1 << 4) * (X[1:-1, 2:] >= X[1:-1, 1:-1]) \
        + (1 << 3) * (X[2:, 2:] >= X[1:-1, 1:-1]) \
        + (1 << 2) * (X[2:, 1:-1] >= X[1:-1, 1:-1]) \
        + (1 << 1) * (X[2:, :-2] >= X[1:-1, 1:-1]) \
        + (1 << 0) * (X[1:-1, :-2] >= X[1:-1, 1:-1])
    return X

def preprocess(vals, zca, sizeOut = 46, show = True):
        if (zca != None):
            X_white = zca.transform(vals)
        else:
            X_white = vals
        sizeIn = int(math.sqrt(len(vals)))
        img = np.reshape(vals, (sizeIn, sizeIn)) / 255.0
        mini = np.percentile(X_white, 0.01)
        maxi = np.percentile(X_white, 99.99)
        X = np.asarray(img)
        X = LBH_Norm(X)

        # img_small = 255 * cv2.resize(dumm, (sizeOut, sizeOut))

        if (show):
            cv2.imshow('Original', cv2.resize(img, (280, 280)))
            cv2.imshow('LocalBinaryHists', cv2.resize(X / 255., (280, 280)))
            cv2.waitKey(1)
        return np.asarray(np.reshape(X, sizeOut ** 2), np.int)

def mask_on_rect(img_face):
    Size_For_Eye_Detection = img_face.shape
    faceCenter = (int(Size_For_Eye_Detection[0] * 0.5), int(Size_For_Eye_Detection[1] * 0.4))
    mask = np.zeros((Size_For_Eye_Detection[0], Size_For_Eye_Detection[1]), np.uint8)
    cv2.ellipse(mask, faceCenter, (int(Size_For_Eye_Detection[0] * 0.30), int(Size_For_Eye_Detection[1] * 0.60)), 0, 0,
                360, 255, -1)
    img_face = np.multiply(mask/255, img_face)
    return img_face

def mask_on_rect2(img_face):
    Size_For_Eye_Detection = img_face.shape
    faceCenter = (int(Size_For_Eye_Detection[0] * 0.5), int(Size_For_Eye_Detection[1] * 0.4))
    mask = np.zeros_like(img_face)
    cv2.ellipse(mask, faceCenter, (int(Size_For_Eye_Detection[0] * 0.30), int(Size_For_Eye_Detection[1] * 0.60)), 0, 0,
                360, 255, -1)
    img_face = np.multiply(mask/255, img_face)
    return img_face


def Mask(vals):
    sizeIn = int(math.sqrt(len(vals)))
    img_face = np.reshape(vals, (sizeIn, sizeIn))
    img_face = mask_on_rect(img_face)
    return np.reshape(img_face, len(vals))

def shared_dataset(data_xy, perm, borrow=True, testnum=0):
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
        #      Testing
        if False:
            import matplotlib.pyplot as plt
            plt.figure(1, figsize=(18, 12))
            plt.subplot(3,2,testnum * 2 + 1)
            plt.hist(data_y)
            plt.title("Ys")
            plt.subplot(3,2,testnum * 2 + 2)
            plt.hist(data_x.reshape(-1))
            plt.title("Xs")
            if (testnum == 2):
                plt.show()

        # When storing data on the GPU it has to be stored as floats
        # therefore we will store the labels as ``floatX`` as well
        # (``shared_y`` does exactly that). But during our computations
        # we need them as ints (we use labels as index, and if they are
        # floats it doesn't make sense) therefore instead of returning
        # ``shared_y`` we will have to cast it to int. This little hack
        # lets ous get around this issue
        return shared_x, T.cast(shared_y, 'int32')

def loadFromCSV(filename):
        y_table = np.zeros(10)
        y_tmp = []
        x_tmp = []
        if (show):
            cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
            cv2.namedWindow('Preprocessed', cv2.WINDOW_NORMAL)
        minV = 1e100
        maxV = -1e100
        with gzip.open(filename) as f:
            reader = csv.reader(f)
            for row in reader:
                y = int(row[0])
                y_tmp.append(y)
                y_table[y] += 1
                vals = np.asarray(row[1:], np.uint8)
                else:
                    preprocessed = preprocess(vals, zca)
                minV = min(min, np.amin(preprocessed))
                maxV = max(max, np.amax(preprocessed))
                x_tmp.append(preprocessed / 255.)
                if (show):
                    n = int(np.sqrt(preprocessed.shape[0]))
                    cv2.imshow('Original', np.reshape(vals/255., (n, n)))
                    cv2.imshow('Preprocessed', np.reshape(preprocessed/255., (n, n)))

        print("  Data Range" + str(minV) + "  " + str(maxV))
        print("  Balance " + str(y_table))
        return (np.asarray(x_tmp, theano.config.floatX), np.asarray(y_tmp, theano.config.floatX))

def giveMeNewTraining():
    y_table = np.zeros(10)
    y_tmp = []
    x_tmp = []
    if (show):
        cv2.namedWindow('Original', cv2.WINDOW_NORMAL)
        cv2.namedWindow('Preprocessed', cv2.WINDOW_NORMAL)
    minV = 1e100
    maxV = -1e100
    c = 0
    with gzip.open(filenameValidation) as f:
        reader = csv.reader(f)
        for row in reader:
            y = int(row[0])
            y_tmp.append(y)
            y_table[y] += 1
            vals = np.asarray(row[1:], np.uint8)
            n = int(np.sqrt(vals.shape[0]))
            img_org = np.reshape(vals/255., (n, n))
            import expandTraining
            img_dist = expandTraining.distorb(img_org)
            img_dist = mask_on_rect2(img_dist)
            x_tmp.append(img_dist.reshape(-1))
            if show:
                cv2.imshow('Original', img_org)
                cv2.imshow('Preprocessed', img_dist)
                cv2.waitKey(1)
            c += 1

    print("  Data Range" + str(minV) + "  " + str(maxV) + " number of training examples " + str(c))
    print("  Balance " + str(y_table))
    train_set = np.asarray(x_tmp, theano.config.floatX), np.asarray(y_tmp, theano.config.floatX)
    return shared_dataset(train_set, np.random.permutation(train_set[1].shape[0]), testnum=2)



# Loads the pictures and creates the data as needed for theano
def load_pictures():
    print theano.config





    test_set = loadFromCSV(filenameTesting)
    print(" Number of test examples [" + str(test_set[1].shape[0]) + "]")
    valid_set = loadFromCSV(filenameValidation)
    print(" Number of validation examples [" + str(valid_set[1].shape[0]) + "]")
    import os.path, time
    print "Test set last modified: %s" % time.ctime(os.path.getmtime(filenameTesting))
    # train_set = loadFromCSV(filenameTraining, zca)
    # print(" Number of training examples [" + str(train_set[1].shape[0]) + "]")
    if (show):
        cv2.destroyAllWindows()
    test_set_x, test_set_y = shared_dataset(test_set, np.random.permutation(test_set[1].shape[0]), testnum=0)
    valid_set_x, valid_set_y = shared_dataset(valid_set, np.random.permutation(valid_set[1].shape[0]), testnum=1)
    #train_set_x, train_set_y = shared_dataset(train_set, np.random.permutation(train_set[1].shape[0]), testnum=2)

    return [(valid_set_x, valid_set_y), (test_set_x, test_set_y)]




if __name__ == "__main__":
    load_pictures()
