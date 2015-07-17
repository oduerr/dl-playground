from lasagne import layers
from lasagne import nonlinearities
from nolearn.lasagne import NeuralNet

__author__ = 'oli'
import os
import Image
import ImageOps
import numpy as np
path = '/Users/oli/Dropbox/Photos/Sample Album/'
imgs = os.listdir(path)


# Creating images (fake only 3 real images) just for demonstration
PIXELS = 96
X = np.zeros((100,3, PIXELS, PIXELS), dtype='float32')
y = np.zeros(100)
for i in range(0,100):
    d = i % 3
    y[i] = d
    img = Image.open(path + imgs[d])
    img = ImageOps.fit(img, (PIXELS, PIXELS), Image.ANTIALIAS)
    img = np.asarray(img, dtype = 'float32') / 255.
    img = img.transpose(2,0,1).reshape(3, PIXELS, PIXELS)
    X[i] = img


net1 = NeuralNet(
    layers=[
        ('input', layers.InputLayer),
        ('hidden1', layers.DenseLayer),
        ('hidden2', layers.DenseLayer),
        ('output', layers.DenseLayer),
        ],
    input_shape=(None, 3, PIXELS, PIXELS), #None in the first axis indicates that the batch size can be set later
    hidden1_num_units=500,
    hidden2_num_units=50,
    output_num_units=10, output_nonlinearity=nonlinearities.softmax,

    # learning rate parameters
    update_learning_rate=0.01,
    update_momentum=0.9,
    regression=False,
    # We only train for 10 epochs
    max_epochs=10,
    verbose=1,

    # Training test-set split
    eval_size = 0.2
        )

X = X.astype(np.float32)
y = y.astype(np.int32)
print(X.shape)
print(y.shape)
net1.fit(X, y)