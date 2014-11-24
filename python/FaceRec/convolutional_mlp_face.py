"""This tutorial introduces the LeNet5 neural network architecture
using Theano.  LeNet5 is a convolutional neural network, good for
classifying images. This tutorial shows how to build the architecture,
and comes with all the hyper-parameters you need to reproduce the
paper's MNIST results.


This implementation simplifies the model in the following ways:

 - LeNetConvPool doesn't implement location-specific gain and bias parameters
 - LeNetConvPool doesn't implement pooling by average, it implements pooling
   by max.
 - Digit classification is implemented with a logistic regression rather than
   an RBF network
 - LeNet5 was not fully-connected convolutions at second layer

References:
 - Y. LeCun, L. Bottou, Y. Bengio and P. Haffner:
   Gradient-Based Learning Applied to Document
   Recognition, Proceedings of the IEEE, 86(11):2278-2324, November 1998.
   http://yann.lecun.com/exdb/publis/pdf/lecun-98.pdf

"""
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
from theano.tensor.shared_randomstreams import RandomStreams

from LogisticRegression import LogisticRegression
from LeNetConvPoolLayer import LeNetConvPoolLayer

from HiddenLayer import HiddenLayer

import pickle
import Utils_dueo

try:
    import PIL.Image as Image
except ImportError:
    import Image

class LeNet5Topology(object):

    def __init__(self):
        self.ishape = (46, 46)      # this is the size of the input image
        self.filter_1 = 5           # Size of first filter
        self.pool_1 = 3             # Size of pooling layer
        self.in_2 = 14              #Input in second layer (layer1)
        self.filter_2 = 5
        self.pool_2 = 2
        self.nkerns = [20,100]
        self.hidden_input = 5*5
        self.numLogisticInput = 200
        self.numLogisticOutput = 6

    def __str__(self):
        return ("Image Shape            " + str(self.ishape[0]) + "x" + str(self.ishape[1]))+ "\n" \
        +("First Filter:          " + str(self.filter_1)) + "\n" \
        +("First Pooling:         " + str(self.pool_1)) + "\n" \
        +("Image Shape (Layer 2)  " + str(self.in_2)) + "\n" \
        +("Second Filter:         " + str(self.filter_2)) + "\n" \
        +("Second Pooling:        " + str(self.pool_2)) + "\n" \
        +("Number of Kernels      " + str(self.nkerns)) + "\n" \
        +("Hidden Input:          " + str(self.hidden_input)) + "\n" \
        +("Logistic Input:        " + str(self.numLogisticInput)) + "\n" \
        +("Logistic Output:       " + str(self.numLogisticOutput))

        #ishape = (28, 28)
        #Orignial Run
        # filter_1 = 5
        # filter_2 = 5
        # in_2 = 12
        # pool_1 = 2
        # pool_2 = 2
        # hidden_input = 4*4
        # numLogisticInput = 200

        # ishape = (28, 28)
        # filter_1 = 5
        # pool_1 = 3
        # in_2 = 8      #Input in second layer (layer1)
        # filter_2 = 3
        # pool_2 = 2
        # hidden_input = 3*3
        # numLogisticInput = 200

class LeNet5State(object):
    """
        The learned state of a LeNet5 (Weights and Biases)
    """
    def __init__(self, topology, convValues, hiddenValues, logRegValues):
        self.topoplogy = topology
        self.convValues = convValues
        self.hiddenValues = hiddenValues
        self.logRegValues = logRegValues



class LeNet5(object):

    def __init__(self, datasets, n_out, topology, nkerns=[20,20], batch_size = 30):
        """
        :param nkerns:
        :return:
        """
        rng = numpy.random.RandomState(23455)
        theano_rng = RandomStreams(numpy.random.randint(2 ** 30))
        # Images for face recognition

        train_set_x, train_set_y = datasets[0]
        valid_set_x, valid_set_y = datasets[1]
        test_set_x, test_set_y = datasets[2]

        # compute number of minibatches for training, validation and testing
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
        n_test_batches = test_set_x.get_value(borrow=True).shape[0]

        n_train_batches /= batch_size
        n_valid_batches /= batch_size
        n_test_batches /= batch_size

        index = T.lscalar()     # index to a [mini]batch
        x = T.matrix('x')       # the data is presented as rasterized images
        y = T.ivector('y')      # the labels are presented as 1D vector of [int] labels

        print '... building the model'
        print 'Number of Kernels' + str(nkerns)

def evaluate_lenet5(topo, learning_rate=0.005, n_epochs=500, datasetName='mnist.pkl.gz',
                    batch_size=4242, createData=False, stateIn = None, stateOut = None):

    global pickle
    rng = numpy.random.RandomState(23455)
    theano_rng = RandomStreams(numpy.random.randint(2 ** 30))

    #Original
    #datasets = load_data(dataset)
    #n_out = 10

    # Images for face recognition
    if (createData):
        import pickle
        import Utils_dueo
        datasets = Utils_dueo.load_pictures()
        pickle.dump(datasets, open( datasetName, "wb" ) ) #Attention y is wrong
        print("Saveing the pickeled data-set")

    #Loading the pickled images
    print("Loading the pickels data-set " + str(datasetName))
    datasets = pickle.load(open(datasetName, "r"))
    n_out = 6
    batch_size = 10
    print("       Learning rate " + str(learning_rate))


    # Images for face recognition
    #train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[0]
    test_set_x, test_set_y = datasets[1]

    # compute number of minibatches for training, validation and testing
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]

    n_valid_batches /= batch_size
    n_test_batches /= batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
    # [int] labels

    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    print 'Number of Kernels' + str(topo.nkerns)


    in_2 = 14      #Input in second layer (layer1)


    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, topo.ishape[0], topo.ishape[1]))

    # Using presistent state from last run
    w0 = w1 = b0 = b1 = wHidden = bHidden = wLogReg = bLogReg = None
    if stateIn is not None:
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
        print("Hallo Gallo")

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
                                image_shape=(batch_size, 1, topo.ishape[0],  topo.ishape[0]),
                                filter_shape=(topo.nkerns[0], 1, topo.filter_1, topo.filter_1),
                                poolsize=(topo.pool_1, topo.pool_1), wOld=w0, bOld=b0)

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
                                image_shape=(batch_size, topo.nkerns[0], topo.in_2, topo.in_2),
                                filter_shape=(topo.nkerns[1], topo.nkerns[0], topo.filter_2, topo.filter_2),
                                poolsize=(topo.pool_2, topo.pool_2), wOld=w1, bOld=b1)

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)

    # Evt. some drop out for the fully connected layer
    # Achtung p=1 entspricht keinem Dropout.
    layer2_input = theano_rng.binomial(size=layer2_input.shape, n=1, p=1 - 0.02) * layer2_input

    layer2 = HiddenLayer(rng, input=layer2_input, n_in=topo.nkerns[1] * topo.hidden_input,
                         n_out=topo.numLogisticInput, activation=T.tanh, Wold = wHidden, bOld = bHidden)

    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=topo.numLogisticInput, n_out=n_out, Wold = wLogReg, bOld=bLogReg )

    # Some regularisation (not for the conv-Kernels)
    L2_sqr = (layer2.W ** 2).sum() + (layer3.W ** 2).sum()

    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y) + 0.001 * L2_sqr

    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], layer3.errors(y),
                                 givens={
                                     x: test_set_x[index * batch_size: (index + 1) * batch_size],
                                     y: test_set_y[index * batch_size: (index + 1) * batch_size]})

    validate_model = theano.function([index], layer3.errors(y),
                                     givens={
                                         x: valid_set_x[index * batch_size: (index + 1) * batch_size],
                                         y: valid_set_y[index * batch_size: (index + 1) * batch_size]})

    # create a list of all model parameters to be fit by gradient descent
    params = layer3.params + layer2.params + layer1.params + layer0.params

    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)

    # train_model is a function that updates the model parameters by
    # SGD Since this model has many parameters, it would be tedious to
    # manually create an update rule for each model parameter. We thus
    # create the updates list by automatically looping over all
    # (params[i],grads[i]) pairs.
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))



    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000 # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
    # found
    improvement_threshold = 0.995  # a relative improvement of this much is
    # considered significant
    # go through this many
    # minibatche before checking the network
    # on the validation set; in this case we
    # check every epoch

    best_params = None
    best_validation_loss = numpy.inf
    best_iter = 0
    test_score = 0.
    start_time = time.clock()

    epoch = 0
    done_looping = False
    epoch_fraction = 0.0
    while (epoch < n_epochs) and (not done_looping):
        # New epoch the training set is disturbed again
        print("  Starting new training epoch")
        print("  Manipulating the training set")
        train_set_x, train_set_y = Utils_dueo.giveMeNewTraining()
        n_train_batches = train_set_x.get_value(borrow=True).shape[0]
        n_train_batches /= batch_size
        validation_frequency = min(n_train_batches, patience / 2)
        print("  Compiling new function")
        learning_rate *= 0.993 #See Paper from Cican
        train_model = theano.function([index], cost, updates=updates,
                                      givens={
                                          x: train_set_x[index * batch_size: (index + 1) * batch_size],
                                          y: train_set_y[index * batch_size: (index + 1) * batch_size]})
        print("  Finished compiling the training set")

        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches): #Alle einmal anfassen
            iter = (epoch - 1) * n_train_batches + minibatch_index
            epoch_fraction +=  1.0 / float(n_train_batches)
            if iter % 100 == 0:
                print 'training @ iter = ', iter, ' epoch_fraction ', epoch_fraction
            cost_ij = train_model(minibatch_index)

            if (iter + 1) % validation_frequency == 0:
                # compute zero-one loss on validation set
                validation_losses = [validate_model(i) for i in xrange(n_valid_batches)]
                this_validation_loss = numpy.mean(validation_losses)
                # test it on the test set
                test_start = time.clock();
                test_losses = [test_model(i) for i in xrange(n_test_batches)]
                dt = time.clock() - test_start
                print'Testing %i faces in %f msec image / sec  %f', batch_size * n_test_batches, dt, dt/(n_test_batches * batch_size)
                test_score = numpy.mean(test_losses)
                print('%i, %f, %f, %f, 0.424242' % (epoch,  this_validation_loss * 100.,test_score * 100., learning_rate))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss * improvement_threshold:
                        patience = max(patience, iter * patience_increase)

                    # save best validation score and iteration number
                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    # # test it on the test set
                    # test_losses = [test_model(i) for i in xrange(n_test_batches)]
                    # test_score = numpy.mean(test_losses)
                    # print(('     epoch %i, minibatch %i/%i, test error of best '
                    #        'model %f %%') %
                    #       (epoch, minibatch_index + 1, n_train_batches,
                    #        test_score * 100.))

                # if (this_validation_loss < 0.02):
                #     learning_rate /= 2
                #     print("Decreased learning rate due to low xval error to " + str(learning_rate))


            if patience <= iter:
                print("--------- Finished Looping ----- earlier ")
                done_looping = True
                break

    end_time = time.clock()
    print('----------  Optimization complete -------------------------')
    print('Res: ', str(topo.nkerns))
    print('Res: ', learning_rate)
    print('Res: Best validation score of %f %% obtained at iteration %i,' \
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('Res: The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
    # Oliver
    if not os.path.isdir("conv_images"):
        os.makedirs("conv_images")
        os.chdir("conv_images")

    # d = layer0.W.get_value() #e.g.  (20, 1, 5, 5) number of filter, num of incomming filters, dim filter
    # for i in range(0, numpy.shape(d)[0]):
    #     dd = d[i][0]
    #     rescaled = (255.0 / dd.max() * (dd - dd.min())).astype(numpy.uint8)
    #     img = Image.fromarray(rescaled)
    #     img.save('filter_l0' + str(i) + '.png')
    #
    # d = layer1.W.get_value() #e.g.  (20, 1, 5, 5) number of filter, num of incomming filters, dim filter
    # for i in range(0, numpy.shape(d)[0]):
    #     dd = d[i][0]
    #     rescaled = (255.0 / dd.max() * (dd - dd.min())).astype(numpy.uint8)
    #     img = Image.fromarray(rescaled)
    #     img.save('filter_l1' + str(i) + '.png')

    state = LeNet5State(topology=topo,
                        convValues = [layer0.getParametersAsValues(), layer1.getParametersAsValues()],
                        hiddenValues = layer2.getParametersAsValues(),
                        logRegValues = layer3.getParametersAsValues())
    print
    if stateOut is not None:
        pickle.dump(state, open(stateOut, 'wb') ) #Attention y is wrong
        print("Saved the pickeled data-set")

    return learning_rate

    ##############################

    #image = Image.fromarray(lay)
    #image.save('samples.png')
    #os.chdir('../')

if __name__ == '__main__':

    topo = LeNet5Topology()
    print(str(topo))

    #import subprocess, time
    #label = subprocess.check_output(['git', 'rev-parse', 'HEAD'])[:-1]
    filename = "Dataset_test_aligned_extended_LBHK100.p"
    import os
    stateIn = None
    state = 'state_lbh_elip_scale_K100'
    if state is not None and os.path.isfile(state):
        stateIn = state
    else:
        stateIn = None


    # Learning and Evaluating leNet
    lr  = 0.1
    stateIn = None
    stateOut = state
    for i in xrange(0,100):
        print(str(lr))
        lr = evaluate_lenet5(topo=topo, learning_rate=lr, datasetName=filename, n_epochs=10, createData=True, stateIn=stateIn, stateOut=stateOut)
        stateIn = stateOut

    # evaluate_lenet5(learning_rate=0.1, datasetName=filename)
    # evaluate_lenet5(learning_rate=1.0, datasetName=filename)
    # evaluate_lenet5(learning_rate=0.5, datasetName=filename)
    # evaluate_lenet5(learning_rate=0.01, datasetName=filename)
    # evaluate_lenet5(learning_rate=0.0001, datasetName=filename) #Best validation score of 23.333333 % obtained at iteration 19950,with test performance 28.666667 %
    # evaluate_lenet5(learning_rate=0.001, datasetName=filename) #<---- Best validation score of 16.666667 % obtained at iteration 4347,with test performance 22.666667

def experiment(state, channel):
    pass
