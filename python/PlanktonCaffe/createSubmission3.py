import sys
import csv,os
import numpy as np
import time
sys.path.append('~/caffe/caffe/python/caffe')
caffe_root = '/home/dueo/caffe/caffe/' 
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2

caffe.set_phase_test()
caffe.set_mode_cpu()
net = caffe.Net('alexnet/train_val3.prototxt', 'alexnet/models/run8/alexnet_128_test_dropout_iter_36000.caffemodel')
inn = caffe.io.load_image('/home/dueo/data_kaggel_bowl/test//1.jpg', color=False)

#TODO substract mean
#TODO test-data augmentation as 

inn_resized = cv2.resize(inn, (99, 99))
inn_batch=np.asarray([[inn_resized]])
#inn_batch = inn_batch.transpose(0,1,2,3)
np.shape(inn_batch)
net.forward_all(data=inn_batch)




