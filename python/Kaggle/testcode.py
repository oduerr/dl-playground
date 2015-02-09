__author__ = 'oli'

import numpy as np

a = [[0.2,0.5,0.3], [0.2,0.5,0.3]]
p = np.asarray(a)
y = np.asarray([0,1])

print([np.arange(y.shape[0]), y])
