from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import BatchIterator
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet
import time


import gzip
import numpy as np
import pickle
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold

PIXELS = 48

class MyNeuralNet(NeuralNet):

    def train_test_split(self, X, y, eval_size):
        #kf = KFold(len(y), 1. / eval_size)
        #train_indices, valid_indices = iter(kf).next()
        train_indices = range(0,192)
        valid_indices = range(192,240)
        X_train, y_train = X[train_indices], y[train_indices]
        X_valid, y_valid = X[valid_indices], y[valid_indices]
        return X_train, X_valid, y_train, y_valid

class SimpleBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        if  not yb == None:
            for i in range(10):
                x,y = np.random.randint(PIXELS/2, 160-PIXELS/2,2)
                retY = yb[:,:,x,y].reshape(len(yb))
                if (retY.max() != 0):
                    break
            #print(i)
            retX = Xb[:,:,(x-PIXELS/2):(x+PIXELS/2),(y-PIXELS/2):(y+PIXELS/2)]
            #print("Made Patches around " + str(x) + "," + str(y) + " width " +  str(retX.shape) + "  " + str(retY.shape))
            return retX,retY#TODO check if x,y are correct
        else:
            return Xb,yb


net1 = MyNeuralNet(
    # Geometrie of the network
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
    output_num_units=10, output_nonlinearity=nonlinearities.softmax,

    # learning rate parameters
    update_learning_rate=0.01,
    update_momentum=0.9,
    regression=False,
    # We only train for 10 epochs
    max_epochs=10,
    verbose=1,

    # Training test-set split
    eval_size = 0.2,

    batch_iterator_train = SimpleBatchIterator(10),
    batch_iterator_test=SimpleBatchIterator(10)
    )




# Setting the new batch iterator
net1.batch_iterator_train = SimpleBatchIterator(batch_size=10)
#net1.batch_iterator_test = SimpleBatchIterator(batch_size=10)


if __name__ == '__main__':
    start = time.time()
    with gzip.open('data/data.pkl.gz', 'rb') as f:
        X,Y = pickle.load(f)
    print ("Loaded data in " + str(time.time() - start))
    print ("   " + str(X.shape) + " y " + str(Y.shape))
    X = X / X.max()
    net1.max_epochs = 2
    net1.fit(X[0:240,:,:,:],Y[0:240,:,:,:]) #Achtung Zahlen sind noch festcodiert
    with open('data/net1.pickle', 'wb') as f:
        pickle.dump(net1, f, -1)

    start = time.time()
    with open('data/net1.pickle', 'rb') as f:
        net_pretrain = pickle.load(f)
    print ("Loaded net in " + str(time.time() - start))

    ddd = net1.predict(X[240:241,:,48:(48+48),48:(48+48)])
    print("Hallo")






