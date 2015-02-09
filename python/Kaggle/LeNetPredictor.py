"""
    Loading as pre-trained LeNet5 for prediction
"""
__author__ = 'oliver'

from convolutional_mlp_plankton import LeNet5State, LeNet5Topology
import LoadPics
from LogisticRegression import LogisticRegression
from LeNetConvPoolLayer import LeNetConvPoolLayer
from HiddenLayer import HiddenLayer

import numpy as np
import theano
from theano import tensor as T
import pickle

import math
import csv
import gzip
import cv2
import csv

class LeNetPredictor(object):

    def __init__(self, stateIn, deepOut = False):
        global pickle

        print("  Loading previous state ...")
        if stateIn.endswith('gz'):
            f = gzip.open(stateIn,'rb')
        else:
            f = open(stateIn, 'r')
        state = pickle.load(f)
        convValues = state.convValues
        w0 = convValues[0][0]
        b0 = convValues[0][1]
        w1 = convValues[1][0]
        b1 = convValues[1][1]
        hiddenVals = state.hiddenValues
        wHidden = hiddenVals[0]
        bHidden = hiddenVals[1]
        logRegValues = state.logRegValues
        wLogReg = logRegValues[0]
        bLogReg = logRegValues[1]
        topo = state.topoplogy
        nkerns = topo.nkerns
        n_out = topo.numLogisticOutput
        print("  Some Values ...")
        print("     Number of Kernels : " + str(nkerns))
        print("     First Kernel w0[0][0] :\n" + str(w0[0][0]))
        print("     bHidden :\n" + str(bHidden))
        print("     bLogReg :\n" + str(bLogReg))
        print("  Building the theano model")
        batch_size = 1

        x = T.matrix('x')   # the data is presented as rasterized images
        layer0_input = x.reshape((batch_size, 1, topo.ishape[0], topo.ishape[1]))
        rng = np.random.RandomState(23455)

        layer0 = LeNetConvPoolLayer(None, input=layer0_input,
                                image_shape=(batch_size, 1, topo.ishape[0],  topo.ishape[0]),
                                filter_shape=(nkerns[0], 1, topo.filter_1, topo.filter_1),
                                poolsize=(topo.pool_1, topo.pool_1), wOld=w0, bOld=b0, deepOut=deepOut)


        layer1 = LeNetConvPoolLayer(None, input=layer0.output,
                                    image_shape=(batch_size, nkerns[0], topo.in_2, topo.in_2),
                                    filter_shape=(nkerns[1], nkerns[0], topo.filter_2, topo.filter_2),
                                    poolsize=(topo.pool_2, topo.pool_2), wOld=w1, bOld=b1, deepOut=deepOut)

        layer2_input = layer1.output.flatten(2)

        layer2 = HiddenLayer(None, input=layer2_input, n_in=nkerns[1] * topo.hidden_input,
                             n_out=topo.numLogisticInput, activation=T.tanh, Wold = wHidden, bOld = bHidden)

        # classify the values of the fully-connected sigmoidal layer
        layer3 = LogisticRegression(input=layer2.output, n_in=topo.numLogisticInput, n_out=n_out, Wold = wLogReg, bOld=bLogReg )

        # create a function to compute the mistakes that are made by the model
        # index = T.lscalar()
        # test_model = theano.function([index], layer3.getProbs(),
        #                              givens={x: test_set_x[index * batch_size: (index + 1) * batch_size]})

        self.predict_model = theano.function([x], layer3.getProbs())

        if (deepOut):
            self.layer0_out = theano.function([x], layer0.output)
            self.layer0_conv= theano.function([x], layer0.conv_out)
            self.layer1_conv= theano.function([x], layer1.conv_out)
            self.layer1_out = theano.function([x], layer1.output)
            self.b0 = b0
            self.b1 = b1
            self.w0 = w0
            self.w1 = w1


    def getPrediction(self, imgAsRow):
        """
            :param imgAsRow: integers in the range [0,255]
            :return:
        """
        values=np.reshape(imgAsRow, (46, 46))
        return (self.predict_model(values))

    def getPool0Out(self, imgAsRow):
        values=np.reshape(imgAsRow, (46, 46))
        return (self.layer0_out(values))

    def getConv0Out(self, imgAsRow):
        values=np.reshape(imgAsRow, (46, 46))
        return (self.layer0_conv(values))

    def getConv1Out(self, imgAsRow):
        values=np.reshape(imgAsRow, (46, 46))
        return (self.layer1_conv(values))

    def getPool1Out(self, imgAsRow):
        values=np.reshape(imgAsRow, (46, 46))
        return (self.layer1_out(values))


if __name__ == "__main__":
    import os
    import Preprocessing
    import time
    if os.path.isfile('plankton.p'):
        stateIn = 'plankton.p'
    else:
        stateIn = None
    pred = LeNetPredictor(stateIn=stateIn)
    print("Loaded Predictor ")


    path_training = "/Users/oli/Proj_Large_Data/kaggle_plankton/train_resized/"
    d = LoadPics.LoadPics(path_training)
    print(d.getNumberOfClassed())


    path = "/Users/oli/Proj_Large_Data/kaggle_plankton/test_resized/"
    files = os.listdir(path)
    c = 0
    fout = open("/Users/oli/Proj_Large_Data/kaggle_plankton/submission.csv", 'w');
    import csv
    w = csv.writer(fout);
    classes = d.getClasses()
    classes.insert(0, 'image')
    w.writerow(classes)
    for fin in files:
        print (fin)
        pics = cv2.imread(path + fin , cv2.CV_LOAD_IMAGE_GRAYSCALE)
        X = np.reshape(pics / 255., len(pics)**2)
        res = pred.getPrediction(X)[0]
        fout.write(fin + ',')
        w.writerow(res)








