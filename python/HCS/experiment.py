__author__ = 'oli'
import gzip
import numpy as np
from sklearn.cross_validation import KFold
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
import time
from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet
import matplotlib.pyplot as plt
import matplotlib.image as imgplot
import numpy as np
from skimage import transform as tf

def load_blob(filename):
    start = time.time()
    npzfile = np.load(filename)
    start = time.time()
    cell_rows = npzfile['arr_0']
    X = npzfile['arr_1']
    Y = npzfile['arr_2']
    print ("Loaded data in " + str(time.time() - start))
    return X,Y, cell_rows

def preprocess(X, Y):
    # Normalization
    Xmean = X.mean(axis = 0)
    XStd = np.sqrt(X.var(axis=0))
    X = (X-Xmean)/(XStd + 0.01)
    Y = np.asarray(Y,dtype='int32')

    # Split
    perm1 = np.random.permutation(len(Y))
    N_split = int(len(Y) * 0.8)
    N_split
    idx_train  = perm1[:N_split]
    idx_test  = perm1[N_split:]

    X_train = X[idx_train,:,:,:]
    Y_train = Y[idx_train]
    X_test = X[idx_test,:,:,:]
    Y_test = Y[idx_test]

     # Permuting the training
    perm = np.random.permutation(len(Y_train))
    return X_train[perm,:,:,:], Y_train[perm], X_test, Y_test

def createSimpleNet(PIXELS):
    net1 = NeuralNet(
    # Geometry of the network
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
            ('dropout4', layers.DropoutLayer),

            ('output', layers.DenseLayer),
            ],
        input_shape=(None, 5, PIXELS, PIXELS), #None in the first axis indicates that the batch size can be set later
        conv1_num_filters=32, conv1_filter_size=(3, 3), pool1_pool_size=(2, 2), #pool_size used to be called ds in old versions of lasagne
        dropout1_p=0.3,

        conv2_num_filters=64, conv2_filter_size=(3, 3), pool2_pool_size=(2, 2),
        dropout2_p=0.3,

        hidden3_num_units=100,
        dropout3_p=0.3,

        hidden4_num_units=20,
        dropout4_p=0.3,

        output_num_units=5, output_nonlinearity=nonlinearities.softmax,

        # learning rate parameters
        update_learning_rate=0.01,
        update_momentum=0.9,
        regression=False,
        # We only train for 10 epochs
        max_epochs=20,
        verbose=1,

        # Training test-set split
        eval_size = 0.2
    )
    return net1

def manipulateTrainingData(Xb, rots):

    retX = np.zeros((Xb.shape[0], Xb.shape[1], Xb.shape[2], Xb.shape[3]), dtype='float32')
    for i in range(len(Xb)):
        rot = rots[np.random.randint(0, len(rots))]

        tf_rotate = tf.SimilarityTransform(rotation=rot)
        shift_y, shift_x = np.array((X.shape[2], X.shape[3])) / 2.
        tf_shift = tf.SimilarityTransform(translation=[-shift_x, -shift_y])
        tf_shift_inv = tf.SimilarityTransform(translation=[shift_x, shift_y])
        tform_rot = (tf_shift + (tf_rotate + tf_shift_inv))

        ## TODO add the transformations
        scale = np.random.uniform(0.9,1.10)
        d = tf.SimilarityTransform(scale=scale, translation=(np.random.randint(5),np.random.randint(5)))
        tform_other = (tform_rot + d)

        for c in range(np.shape(X)[1]):
            maxAbs = 256.0;np.max(np.abs(Xb[i,c,:,:]))
            # Needs at lease 0.11.3
            retX[i,c,:,:] = tf.warp(Xb[i,c,:,:], tform_other, preserve_range = True) # "Float Images" are only allowed to have values between -1 and 1
    return retX

if __name__ == '__main__':
    filename = 'HCS_72x72_small_32.npz'
    X,Y,cell_rows = load_blob(filename)
    print('dim Y {0}, dim X {1}, type X {2}'.format(np.shape(Y), np.shape(X), type(X)))

    X_train, Y_train, X_test, Y_test = preprocess(X, Y)

    if False:
        Xb = np.copy(X[0:10,:,:,:])
        Xb = manipulateTrainingData(Xb, rots=np.deg2rad(range(0,359)))
        fig = plt.figure(figsize=(10,10))
        for i in range(5):
            fig.add_subplot(6,6,2*i+1,xticks=[], yticks=[])
            plt.imshow(X[i,0,:,:], cmap=plt.get_cmap('cubehelix'))
            fig.add_subplot(6,6,2*i+2,xticks=[], yticks=[])
            plt.imshow(Xb[i,0,:,:], cmap=plt.get_cmap('cubehelix'))
            print('before {0} after {1}'.format(np.mean(X[i,1,:,:]), np.mean(Xb[i,1,:,:])))

        fig.show()
        fig.waitforbuttonpress()
    #net = createSimpleNet(72)
    #net.fit(X_train, Y_train)

