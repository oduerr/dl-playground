from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import BatchIterator
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet
import time


import gzip
import numpy as np

################ Pickle with an updated recursion limit
import pickle
import sys
sys.setrecursionlimit(10000)

PIXELS = 48
CLASSES = 2
DEBUG = True

class MyNeuralNet(NeuralNet):

    def train_test_split(self, X, y, eval_size):
        print("Doing the Training Testing Split")
        train_indices = range(0,192)
        valid_indices = range(192,240)
        #valid_indices = range(0,80)
        #train_indices = range(80,240)
        X_train, y_train = X[train_indices], y[train_indices]
        X_valid, y_valid = X[valid_indices], y[valid_indices]
        return X_train, X_valid, y_train, y_valid


#   A BatchIterator which cut's part of the training set
class SegmentationBatchIterator(BatchIterator):

    def __init__(self, batch_size, nClasses, nPixels):  #Python is so ugly!
        super(self.__class__, self).__init__(batch_size) # <----      Das sieht doch zum kotzen aus!
        self.nClasses = nClasses
        self.nPixels = nPixels
        # We should initialize with 0 but then there is a possibility of division by zero
        # but we don't make a big mistake by initialising with 1
        self.freq = np.ones(nClasses, dtype=np.int64)
        if DEBUG: print("------------   Constructor has been called --------------------- ")


    def transform(self, Xb, yb):
        if  not yb == None: #Training or Validation
            retYs = np.zeros(len(yb), dtype='uint8')
            retX = np.zeros((Xb.shape[0], Xb.shape[1], PIXELS, PIXELS), dtype='float32')
            for b in range(len(yb)):
                for i in range(1000): #Trying to sample 1000 times.
                    x,y = np.random.randint(PIXELS/2, 160-PIXELS/2,2)
                    example = np.random.randint(len(yb)) #Choose a random example from the training set
                    rety = yb[example,:,x,y]
                    if (rety > 0): #Only for one class
                      rety = 1
                    rel = float(self.freq[rety]) / self.freq.sum()
                    ap = (1.0 - rel + 0.2)**8
                    goal = 1.0 / self.nClasses
                    if (rel < goal + 0.05):
                        break
                self.freq[rety] = self.freq[rety] + 1
                retYs[example] = rety
                retX[example,:,:,:] = Xb[example,:,(x-PIXELS/2):(x+PIXELS/2),(y-PIXELS/2):(y+PIXELS/2)]
            if DEBUG: print(str(i) + " " + str(retYs.mean()) + " batchsize " + str(retX.shape) + "   freq= " + str(self.freq) + " sum " + str(self.freq.sum()))
            #print("Made Patches around " + str(x) + "," + str(y) + " width " +  str(retX.shape) + "  " + str(retY.shape))

            ##### Plotting
            #import cv2
            #cv2.imshow('Test', retX[0,0,:,:])
            #cv2.waitKey(100)
            return retX,retYs#TODO check if x,y are correct
        else:
            if DEBUG: print("Made Patches around ")
            return Xb,yb



net1 = MyNeuralNet(
    # Geometrie of the network
    layers=[
        ('input', layers.InputLayer),
        ('conv1', layers.Conv2DLayer),
        ('pool1', layers.MaxPool2DLayer),
        ('dropout1', layers.DropoutLayer),
        ('conv2', layers.Conv2DLayer),
        ('pool2', layers.MaxPool2DLayer),
        ('dropout2', layers.DropoutLayer),
        ('hidden3', layers.DenseLayer),
        ('dropout3', layers.DropoutLayer),
        ('hidden4', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 1, PIXELS, PIXELS),
    conv1_num_filters=32, conv1_filter_size=(5, 5), pool1_ds=(2, 2), dropout1_p=0.2,
    conv2_num_filters=64, conv2_filter_size=(3, 3), pool2_ds=(2, 2), dropout2_p=0.2,
    hidden3_num_units=256, dropout3_p=0.5,
    hidden4_num_units=128,
    output_num_units=CLASSES, output_nonlinearity=nonlinearities.softmax,

    # learning rate parameters
    update_learning_rate=0.001,
    update_momentum=0.09,
    regression=False,
    # We only train for 10 epochs
    max_epochs=10,
    verbose=1,

    # Training test-set split
    eval_size = 0.2,

    batch_iterator_train = SegmentationBatchIterator(128, CLASSES, PIXELS),
    batch_iterator_test=SegmentationBatchIterator(128, CLASSES, PIXELS)
    )




# Setting the new batch iterator
#net1.batch_iterator_train = SimpleBatchIterator(batch_size=10)
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
    ddd = net1.predict_proba(X[240:241,:,48:(48+48),48:(48+48)])
    print("Hallo")






