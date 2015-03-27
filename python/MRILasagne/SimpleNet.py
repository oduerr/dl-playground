__author__ = 'oli'

from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import BatchIterator
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet
from dataloading import *

sl = loadSimpleData()
X,y = sl.load2d('data/data56.pkl')
PIXELS = sl.PIXELS

# for i in range(10):
#     import cv2
#     ddd=X[i,0,:,:]
#     cv2.imshow('Test', cv2.resize(ddd, (PIXELS*10, PIXELS*10), interpolation =cv2.INTER_NEAREST))
#     cv2.waitKey(1000)

print("Loaded data")

net1 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('hidden4', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, PIXELS, PIXELS),
    conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_ds=(2, 2),
    conv2_num_filters=64, conv2_filter_size=(2, 2), pool2_ds=(2, 2),
    hidden4_num_units=500,
    output_num_units=2, output_nonlinearity=nonlinearities.softmax,

    update_learning_rate=0.01,
    update_momentum=0.9,

    regression=False,
    max_epochs=1000,
    verbose=1,
    )

class SimpleBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        Xb, yb = super(SimpleBatchIterator, self).transform(Xb, yb)
        # The 'incomming' and outcomming shape is (batchsize, 1, 28, 28)


        return Xb[:,:,::-1,:], yb #<--- Here we do the flipping

net1.batch_iterator_train = SimpleBatchIterator(batch_size=128)
net1.fit(X, y)





