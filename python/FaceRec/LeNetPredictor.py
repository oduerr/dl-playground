"""
    Loading as pre-trained LeNet5 for prediction
"""
__author__ = 'oliver'

from convolutional_mlp_face import LeNet5State, LeNet5Topology
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

class LeNetPredictor(object):

    def __init__(self, stateIn, deepOut = False):
        global pickle

        print("  Loading previous state ...")
        state = pickle.load(open(stateIn, "r"))
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
        print("     Number of Kernels" + str(nkerns))
        print("     First Kernel w0[0][0]" + str(w0[0][0]))




        print("  Building the theano model")
        batch_size = 1

        x = T.matrix('x')   # the data is presented as rasterized images
        layer0_input = x.reshape((batch_size, 1, topo.ishape[0], topo.ishape[1]))
        rng = np.random.RandomState(23455)

        layer0 = LeNetConvPoolLayer(None, input=layer0_input,
                                image_shape=(batch_size, 1, topo.ishape[0],  topo.ishape[0]),
                                filter_shape=(nkerns[0], 1, topo.filter_1, topo.filter_1),
                                poolsize=(topo.pool_1, topo.pool_1), wOld=w0, bOld=b0)


        layer1 = LeNetConvPoolLayer(None, input=layer0.output,
                                    image_shape=(batch_size, nkerns[0], topo.in_2, topo.in_2),
                                    filter_shape=(nkerns[1], nkerns[0], topo.filter_2, topo.filter_2),
                                    poolsize=(topo.pool_2, topo.pool_2), wOld=w1, bOld=b1)

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

        if (deepOut):
            self.layer1_out = theano.function([x], layer1.output)


        # for i in xrange(5):
        #     res = test_model(i)
        #     print("Hallo" + str(res / res.sum()) + str(predict_model))
        #     # xIn = {layer0_input : test_set_x[i].reshape((batch_size, 1, topo.ishape[0], topo.ishape[1]))}
        #     # xIn = {x : test_set_x[i]}
        #     values=numpy.zeros((46, 46), dtype=theano.config.floatX)
        #     res2 = predict_model(values)
        #     print("Gallo" + str(res / res.sum()) + str(predict_model))


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

    def getPool1Out(self, imgAsRow):
        values=np.reshape(imgAsRow, (46, 46))
        return (self.layer1_out(values))


if __name__ == "__main__":
    import os
    import Utils_dueo
    import time
    if os.path.isfile('state.p'):
        stateIn = 'state.p'
    else:
        stateIn = None
    pred = LeNetPredictor(stateIn=stateIn)
    print("Loaded Predictor ")
    filenameTesting    = "../../data/testing_48x48_aligned_large.p_R.csv.gz"
    totTime = 0
    count = 0
    show = True
    ok = 0
    preds = 0
    import matplotlib.pyplot as plt
    plt.ion()
    pos = np.arange(6)+.5
    with gzip.open(filenameTesting) as f:
        reader = csv.reader(f)
        for row in reader:
            truePerson = int(row[0])
            vals = np.asarray(row[1:], np.float)
            start = time.time()
            preprocessed = Utils_dueo.preprocess(vals, None, 46, show)
            res = pred.getPrediction(preprocessed / 255.)
            totTime += time.time() - start
            predPerson = int(res.argmax())
            plt.clf()
            plt.yticks(pos, ('Dejan', 'Diego', 'Martin', 'Oliver', 'Rebekka', 'Ruedi'))
            predPValue = res[0][res.argmax()]
            if (predPValue > 0.9):
                preds += 1
            if predPerson == truePerson:
                if (predPValue > 0.9):
                    ok += 1
                col = 'g'
            else:
                col = 'r'
            plt.barh(pos, np.asarray(res[0], dtype = float), align='center', color = col)
            plt.draw()
            plt.title(" allOK " + str(ok))
            time.sleep(0.01)
            if predPerson != truePerson:
                time.sleep(0.01)
            print(str(truePerson) + " " + str(predPerson) + str(predPValue))
            count += 1
    plt.show()
    print("Total Time " + str(totTime) + "sec for " + str(count) + " faces. Accuracy " + str((1. * ok) / preds)
          + " predicted " + str(preds))




