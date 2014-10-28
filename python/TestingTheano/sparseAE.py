"""
 Starting from the dA of the theano tutorial, this tries to build a sparse autoencoder
"""

import os
import sys
import time

import numpy

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams

from logistic_sgd import load_data
from utils import tile_raster_images

try:
    import PIL.Image as Image
except ImportError:
    import Image

#theano.config.compute_test_value = 'warn'

# start-snippet-1
class dA(object):


    def __init__(
        self,
        numpy_rng,
        theano_rng=None,
        input=None,
        n_visible=784,
        n_hidden=500,
        W=None,
        bhid=None,
        bvis=None
    ):
        """
        Initialize the dA class by specifying the number of visible units (the
        dimension d of the input ), the number of hidden units ( the dimension
        d' of the latent or hidden space ) and the corruption level. The
        constructor also receives symbolic variables for the input, weights and
        bias. Such a symbolic variables are useful when, for example the input
        is the result of some computations, or when weights are shared between
        the dA and an MLP layer. When dealing with SdAs this always happens,
        the dA on layer 2 gets as input the output of the dA on layer 1,
        and the weights of the dA are used in the second stage of training
        to construct an MLP.

        :type numpy_rng: numpy.random.RandomState
        :param numpy_rng: number random generator used to generate weights

        :type theano_rng: theano.tensor.shared_randomstreams.RandomStreams
        :param theano_rng: Theano random generator; if None is given one is
                     generated based on a seed drawn from `rng`

        :type input: theano.tensor.TensorType
        :param input: a symbolic description of the input or None for
                      standalone dA

        :type n_visible: int
        :param n_visible: number of visible units

        :type n_hidden: int
        :param n_hidden:  number of hidden units

        :type W: theano.tensor.TensorType
        :param W: Theano variable pointing to a set of weights that should be
                  shared belong the dA and another architecture; if dA should
                  be standalone set this to None

        :type bhid: theano.tensor.TensorType
        :param bhid: Theano variable pointing to a set of biases values (for
                     hidden units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None

        :type bvis: theano.tensor.TensorType
        :param bvis: Theano variable pointing to a set of biases values (for
                     visible units) that should be shared belong dA and another
                     architecture; if dA should be standalone set this to None


        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden

        # create a Theano random generator that gives symbolic random values
        if not theano_rng:
            theano_rng = RandomStreams(numpy_rng.randint(2 ** 30))

        # note : W' was written as `W_prime` and b' as `b_prime`
        if not W:
            # W is initialized with `initial_W` which is uniformely sampled
            # from -4*sqrt(6./(n_visible+n_hidden)) and
            # 4*sqrt(6./(n_hidden+n_visible))the output of uniform if
            # converted using asarray to dtype
            # theano.config.floatX so that the code is runable on GPU
            initial_W = numpy.asarray(
                numpy_rng.uniform(
                    low=-4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    high=4 * numpy.sqrt(6. / (n_hidden + n_visible)),
                    size=(n_visible, n_hidden)
                ),
                dtype=theano.config.floatX
            )
            W = theano.shared(value=initial_W, name='W', borrow=True)

        if not bvis:
            bvis = theano.shared(
                value=numpy.zeros(
                    n_visible,
                    dtype=theano.config.floatX
                ),
                borrow=True
            )

        if not bhid:
            bhid = theano.shared(
                value=numpy.zeros(
                    n_hidden,
                    dtype=theano.config.floatX
                ),
                name='b',
                borrow=True
            )

        self.W = W
        # b corresponds to the bias of the hidden
        self.b = bhid
        # b_prime corresponds to the bias of the visible
        self.b_prime = bvis
        # tied weights, therefore W_prime is W transpose
        self.W_prime = self.W.T
        self.theano_rng = theano_rng
        # if no input is given, generate a variable representing the input
        if input is None:
            # we use a matrix because we expect a minibatch of several
            # examples, each example being a row
            self.x = T.dmatrix(name='input')
        else:
            self.x = input

        self.params = [self.W, self.b, self.b_prime]
    # end-snippet-1

    def get_hidden_values(self, input):
        """ Computes the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(input, self.W) + self.b)

    def get_reconstructed_input(self, hidden):
        """Computes the reconstructed input given the values of the hidden layer """
        return T.nnet.sigmoid(T.dot(hidden, self.W_prime) + self.b_prime)

    def get_cost_updates(self, corruption_level, learning_rate):
        """ This function computes the cost and the updates for one trainng
        step of the dA """

        y = self.get_hidden_values(self.x)
        z = self.get_reconstructed_input(y)
        # note : we sum over the size of a datapoint; if we are using
        #        minibatches, L will be a vector, with one entry per
        #        example in minibatch
        L = - T.sum(self.x * T.log(z) + (1 - self.x) * T.log(1 - z), axis=1)
        # note : L is now a vector, where each element is the
        #        cross-entropy cost of the reconstruction of the
        #        corresponding example of the minibatch. We need to
        #        compute the average of all these to get the cost of
        #        the minibatch


        ########################################################
        #  dueo begin Penalty for sparse autoencoder
        # dueo testing theano
        # x  = T.matrix('x')
        # # provide Theano with a default test-value
        # x.tag.test_value = numpy.random.rand(5, 784)
        #y = shared(numpy.random.randn(3,4))


        rho = 0.1
        beta = 10
        rho_hat = T.mean(y, axis=1)
        pen = rho * T.log(rho) - rho * T.log(rho_hat) + (1-rho) * T.log(1 - rho) - (1-rho) * T.log(1 - rho_hat)
        cost = T.mean(L) + beta * T.sum(pen)


        # end
        #######################################################################
        #cost = T.mean(L)

        # compute the gradients of the cost of the `dA` with respect
        # to its parameters
        gparams = T.grad(cost, self.params)
        # generate the list of updates
        updates = [
            (param, param - learning_rate * gparam)
            for param, gparam in zip(self.params, gparams)
        ]

        return (cost, updates)


def test_dA(learning_rate=0.1, training_epochs=150,
            dataset='mnist.pkl.gz',
            batch_size=20, output_folder='sparseAE_plots'):

    """
    This demo is tested on MNIST

    :type learning_rate: float
    :param learning_rate: learning rate used for training the DeNosing
                          AutoEncoder

    :type training_epochs: int
    :param training_epochs: number of epochs used for training

    :type dataset: string
    :param dataset: path to the picked dataset

    """
    datasets = load_data(dataset)
    train_set_x, train_set_y = datasets[0]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0] / batch_size

    # allocate symbolic variables for the data
    index = T.lscalar()    # index to a [mini]batch
    x = T.matrix('x')  # the data is presented as rasterized images

    if not os.path.isdir(output_folder):
        os.makedirs(output_folder)
    os.chdir(output_folder)
    ####################################
    # BUILDING THE MODEL NO CORRUPTION #
    ####################################

    rng = numpy.random.RandomState(123)
    theano_rng = RandomStreams(rng.randint(2 ** 30))

    da = dA(
        numpy_rng=rng,
        theano_rng=theano_rng,
        input=x,
        n_visible=28 * 28,
        n_hidden=14*14     # See Ufldl Tutorial Exercise: Vectorization
    )

    cost, updates = da.get_cost_updates(
        corruption_level=0.,
        learning_rate=learning_rate
    )

    train_da = theano.function(
        [index],
        cost,
        updates=updates,
        givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size]
        }
    )

    start_time = time.clock()

    ############
    # TRAINING #
    ############

    # go through training epochs
    for epoch in xrange(training_epochs):
        # go through trainng set
        c = []
        for batch_index in xrange(n_train_batches):
            c.append(train_da(batch_index))
        weights = da.W.get_value(borrow=True)
        wT = weights.T
        #dueo
        d = numpy.zeros(14*14)
        for i in range(0,14*14):
             d[i] = numpy.sqrt(sum(wT[i,]**2))
        weights1 = numpy.divide(weights, d)
        #end
        image = Image.fromarray(
            tile_raster_images(X=weights1.T,
                               img_shape=(28, 28), tile_shape=(14, 14),
                               tile_spacing=(1, 1)))
        image.save('filters_sparse_autoencoder_%d.png' % epoch)
        print 'Training epoch %d, cost ' % epoch, numpy.mean(c)

    end_time = time.clock()

    training_time = (end_time - start_time)

    print >> sys.stderr, ('The no corruption code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((training_time) / 60.))

    os.chdir('../')


if __name__ == '__main__':
    test_dA()
