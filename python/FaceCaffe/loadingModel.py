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


if __name__ == "__main__":
  caffe.set_mode_cpu()
  caffe.set_phase_test()
  #caffe.set_phase_train()
  #net = caffe.Net('lenet_train_test_files.prototxt', 'snapshots/lenet_iter_10000.caffemodel')
  net = caffe.Net('lenet_train_test_files.prototxt', 'snapshots/lenet_iter_2000.caffemodel')
  #net = caffe.Net('lenet_enhanced.prototxt', 'snapshots/lenet25Feb_iter_25000.caffemodel')
  start = time.time()
  res = net.forward() # this will load the next mini-batch as defined in the net (rewinds)
  logloss = res['loss'][0][0][0][0]
  print(res)
  print ("Time for a single forward step " + str((time.time() - start)))
  preds = net.blobs['ip2'].data 
  batchSize = np.shape(preds)[0]
  yTrues = np.reshape(net.blobs['label'].data, batchSize).astype(int) #True Labels (passed from the data layer)
  sumLogLoss = 0
  acc = 0.0
  for i,yTrue in enumerate(yTrues):
    pred = np.reshape(preds[i], 6) #Output of the final layer (no activation function)
    prob = np.exp(pred)/np.sum(np.exp(pred)) #Calculate the activation function
    print(str(i) + " " + str(prob[yTrue]) + " yTrue " + str(yTrue) + " pred " + str(np.argmax(prob)) + " prob Max " + str(prob[np.argmax(prob)]) )
    sumLogLoss -= np.log(prob[yTrue])
    acc += (np.argmax(prob) == yTrue)
  print("Calculated  logloss()" + str(sumLogLoss/batchSize) + "  acc=" + str(acc / batchSize) + " logloss(caffe layer)=" + str(logloss))
