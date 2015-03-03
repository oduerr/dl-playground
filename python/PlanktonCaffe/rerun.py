import sys
import csv,os
import numpy as np
import time
sys.path.append('~/caffe/caffe/python/caffe')
caffe_root = '/home/dueo/caffe/caffe/' 
sys.path.insert(0, caffe_root + 'python')
import caffe

predLayerName = 'fc8' #Output of the last inner product layer
description = 'train_val.prototxt'
learnedModel = 'models/alexnet_128_test_iter_2900.caffemodel'

caffe.set_mode_gpu()
caffe.set_phase_train() #TODO delete
net = caffe.Net(description, learnedModel)





solver = caffe.SGDSolver('solver.auto.prototxt')  
solver.net.copy_from(net)
    
      
