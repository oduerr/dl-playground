import sys
import csv,os
import numpy as np
import time
sys.path.append('~/caffe/caffe/python/caffe')
caffe_root = '/home/dueo/caffe/caffe/' 
sys.path.insert(0, caffe_root + 'python')
import caffe

predLayerName = 'fc8' #Output of the last inner product layer
description = 'val_kaggle.prototxt'
learnedModel = 'models/alexnet_train_iter_5000.caffemodel'
sampleSub    ='../sampleSubmission.csv.head.csv' #We just need the header
submissionName = 'submission.txt'

if __name__ == "__main__":
  # Getting the names of the validation set
  names = []
  with open('../test_kaggle_full.txt', 'rb') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
        names.append(row[0].split("//")[1]) #This file contains 130'000 lines like: /home/dueo/data_kaggel_bowl/test//1.jpg 42
  
  # Getting the header
  fc = csv.reader(file(sampleSub))
  fst =  fc.next()
  # Opening the submission file
  fout = open(submissionName, 'w');
  w = csv.writer(fout);
  w.writerow(fst)
  
  caffe.set_mode_gpu()
  caffe.set_phase_test()
  # Taken from: https://github.com/BVLC/caffe/issues/1774
  net = caffe.Net(description, learnedModel)
  read = 0
  while(read < len(names)):
    start = time.time()
    res = net.forward() # this will load the next mini-batch as defined in the net (rewinds)
    print ("Time for getting the batch " + str((time.time() - start)) + " " + str(read))
    preds = net.blobs[predLayerName].data 
    batchSize = np.shape(preds)[0]
    for i in range(0,batchSize):
      #pred = np.reshape(preds[i],1000)[0:121] #Todo change to 121
      pred = np.reshape(preds[i],121)
      prob = np.exp(pred)/np.sum(np.exp(pred))
      s = names[read]
      for r in prob:
          s = s + ',' + "{0:.4}".format(r)
      fout.write(s + '\n')
      read += 1
      if (read >= len(names)):
        break
  
    
      
