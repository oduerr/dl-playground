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
import cPickle
import gzip
import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

from logistic_sgd import LogisticRegression, load_data
from mlp import HiddenLayer

try:
    import PIL.Image as Image
except ImportError:
    import Image


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = numpy.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /
                   numpy.prod(poolsize))
        # initialize weights with random weights
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(numpy.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)

        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))

        # store parameters of this layer
        self.params = [self.W, self.b]


def evaluate_lenet5(learning_rate=0.005, n_epochs=500,
                    datasetName='mnist.pkl.gz',
                    nkerns=[20, 50], batch_size=4242, createData=False, label = None):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """

    rng = numpy.random.RandomState(23455)

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
    import pickle
    print("Loading the pickels data-set " + str(datasetName))
    datasets = pickle.load(open(datasetName, "r"))
    n_out = 6
    batch_size = 30
    print("       Learning rate " + str(learning_rate))


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

    # allocate symbolic variables for the data
    index = T.lscalar()  # index to a [mini]batch
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')  # the labels are presented as 1D vector of
                        # [int] labels

    ishape = (28, 28)  # this is the size of MNIST images



    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    print 'Number of Kernels' + str(nkerns)


    #Orignial Run
    # filter_1 = 5
    # filter_2 = 5
    # in_2 = 12
    # pool_1 = 2
    # pool_2 = 2
    # hidden_input = 4*4
    # numLogisticInput = 200
    
    filter_1 = 5
    pool_1 = 3
    in_2 = 8      #Input in second layer (layer1)
    filter_2 = 3
    pool_2 = 2
    hidden_input = 3*3
    numLogisticInput = 200
    
    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, ishape[0], ishape[1]))

    # Construct the first convolutional pooling layer:
    # filtering reduces the image size to (28-5+1,28-5+1)=(24,24)
    # maxpooling reduces this further to (24/2,24/2) = (12,12)
    # 4D output tensor is thus of shape (batch_size,nkerns[0],12,12)
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, ishape[0], ishape[0]),
            filter_shape=(nkerns[0], 1, filter_1, filter_1), poolsize=(pool_1, pool_1))

    # Construct the second convolutional pooling layer
    # filtering reduces the image size to (12-5+1,12-5+1)=(8,8)
    # maxpooling reduces this further to (8/2,8/2) = (4,4)
    # 4D output tensor is thus of shape (nkerns[0],nkerns[1],4,4)
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns[0], in_2, in_2),
            filter_shape=(nkerns[1], nkerns[0], filter_2, filter_2), poolsize=(pool_2, pool_2))

    # the HiddenLayer being fully-connected, it operates on 2D matrices of
    # shape (batch_size,num_pixels) (i.e matrix of rasterized images).
    # This will generate a matrix of shape (20,32*4*4) = (20,512)
    layer2_input = layer1.output.flatten(2)


    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns[1] * hidden_input,
                         n_out=numLogisticInput, activation=T.tanh)

    layer25 = HiddenLayer(rng, input=layer2.output, n_in=numLogisticInput,
                         n_out=numLogisticInput, activation=T.tanh)


    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer25.output, n_in=numLogisticInput, n_out=n_out)

    #L1 = abs(layer2.W).sum() + abs(layer3.W).sum()
    L2_sqr = (layer2.W ** 2).sum() + (layer3.W ** 2).sum() +  (layer25.W ** 2).sum()


    # the cost we minimize during training is the NLL of the model
    cost = layer3.negative_log_likelihood(y) + 0.1 * L2_sqr

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
    params = layer3.params + layer25.params + layer2.params + layer1.params + layer0.params

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

    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters
    patience = 10000  # look as this many examples regardless
    patience_increase = 2  # wait this much longer when a new best is
                           # found
    improvement_threshold = 0.995  # a relative improvement of this much is
                                   # considered significant
    validation_frequency = min(n_train_batches, patience / 2)
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
                test_losses = [test_model(i) for i in xrange(n_test_batches)]
                test_score = numpy.mean(test_losses)
                print('%i, %f, %f' % (epoch,  this_validation_loss * 100.,test_score * 100.))

                # if we got the best validation score until now
                if this_validation_loss < best_validation_loss:

                    #improve patience if loss improvement is good enough
                    if this_validation_loss < best_validation_loss *  \
                       improvement_threshold:
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

            if patience <= iter:
                done_looping = True
                break

    end_time = time.clock()
    print('----------  Optimization complete -------------------------')
    print('Res: ', str(nkerns))
    print('Res: ', learning_rate)
    print('Res: Best validation score of %f %% obtained at iteration %i,'\
          'with test performance %f %%' %
          (best_validation_loss * 100., best_iter + 1, test_score * 100.))
    print('Res: The code for file ' + os.path.split(__file__)[1] + ' ran for %.2fm' % ((end_time - start_time) / 60.))
    # Oliver
    if not os.path.isdir("conv_images"):
        os.makedirs("conv_images")
        os.chdir("conv_images")

    d = layer0.W.get_value() #e.g.  (20, 1, 5, 5) number of filter, num of incomming filters, dim filter
    for i in range(0, numpy.shape(d)[0]):
        dd = d[i][0]
        rescaled = (255.0 / dd.max() * (dd - dd.min())).astype(numpy.uint8)
        img = Image.fromarray(rescaled)
        img.save('filter_' + str(i) + '.png')

    #image = Image.fromarray(lay)
    #image.save('samples.png')
    #os.chdir('../')

if __name__ == '__main__':
    #import subprocess, time
    #label = subprocess.check_output(['git', 'rev-parse', 'HEAD'])[:-1]
    filename = "Dataset_test_aligned_extended_LBH.p"
    evaluate_lenet5(learning_rate=0.1, datasetName=filename, n_epochs=20, createData=False)
    evaluate_lenet5(learning_rate=0.1, datasetName=filename)
    evaluate_lenet5(learning_rate=1.0, datasetName=filename)
    evaluate_lenet5(learning_rate=0.5, datasetName=filename)
    evaluate_lenet5(learning_rate=0.01, datasetName=filename)
    evaluate_lenet5(learning_rate=0.0001, datasetName=filename) #Best validation score of 23.333333 % obtained at iteration 19950,with test performance 28.666667 %
    evaluate_lenet5(learning_rate=0.001, datasetName=filename) #<---- Best validation score of 16.666667 % obtained at iteration 4347,with test performance 22.666667

def experiment(state, channel):
    pass
