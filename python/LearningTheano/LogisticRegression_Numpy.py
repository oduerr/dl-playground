__author__ = 'oli'


# Trying to understand the extreme compact notation of the loglikelihood in theano

#-T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])

import numpy as np

# 3 Output Classes
p_y_given_x = np.asmatrix([[1e-1,1e-1, 8e-2], [1e-1,8e-2, 1e-1]])


y = np.asarray((0,1)) # The training set of the minibatch
-np.mean(np.log(p_y_given_x)[np.arange(y.shape[0]), y])

print("Hallo")