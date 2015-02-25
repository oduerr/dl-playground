__author__ = 'oli'
import sys
import csv,os
import numpy as np
import time
sys.path.append('~/caffe/caffe/python/caffe')
caffe_root = '/home/dueo/caffe/caffe/' 
sys.path.insert(0, caffe_root + 'python')
import caffe
print("Finshed imports")

# For google net
predLayerName = 'loss3/classifier' #For the google-net
logLossName = 'loss3/loss3' 
description = 'googlenet/train_val.prototxt'
learnedModel = 'googlenet/models/googlenet_quick_iter_160000.caffemodel'

#predLayerName = 'loss3/classifier' #For the google-net
#logLossName = 'ip2' 
#description = 'lenet/lenet_train_test_files.prototxt'
#learnedModel = 'lenet/model/lenet60_iter_100000.caffemodel '


if __name__ == "__main__":
  caffe.set_mode_cpu()
  caffe.set_phase_test()
  # Taken from:https://github.com/BVLC/caffe/issues/1774
  net = caffe.Net(description, learnedModel)
  start = time.time()
  res = net.forward() # this will load the next mini-batch as defined in the net (rewinds)
  logloss = res[logLossName][0][0][0][0]
  # To create images
  if False:
    import cv2
    dat = net.blobs['data'].data
  print(res)
  print ("Time for a single forward step " + str((time.time() - start)))
  preds = net.blobs[predLayerName].data 
  batchSize = np.shape(preds)[0]
  yTrues = np.reshape(net.blobs['label'].data, batchSize).astype(int)
  summe = 0
  acc = 0
  for i,yTrue in enumerate(yTrues):
    pred = np.reshape(preds[i],121)
    prob = np.exp(pred)/np.sum(np.exp(pred))
    print(str(i) + " " + str(prob[yTrue]) + " yTrue " + str(yTrue) + " pred " + str(np.argmax(prob)))
    summe -= np.log(prob[yTrue])
    acc += (np.argmax(prob) == yTrue)
  print("Hallo   logloss()" + str(summe/batchSize) + "  acc=" + str(acc / batchSize) + " logloss(caffe)=" + str(logloss))
