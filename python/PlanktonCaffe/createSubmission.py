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
  caffe.set_mode_gpu()
  fc = csv.reader(file('sampleSubmission.csv.head.csv'))
  fst =  fc.next()
  fout = open('submission_caffeLeNet.txt', 'w');
  w = csv.writer(fout);
  w.writerow(fst)
  head = fc.next()[1:]
  path = '/home/dueo/data_kaggel_bowl/test_resized/'
  files = os.listdir(path)
  try:
      files.remove('.DS_Store')
  except:
      pass
  print("Read " + str(len(files)) + " files to be classified")
  net = caffe.Classifier('lenet/lenet_deploy.prototxt', 'lenet/model/lenet60_iter_100000.caffemodel', image_dims=(46, 46))
  #net = caffe.Classifier('googlenet/deploy.prototxt', 'googlenet/models/googlenet_quick_iter_160000.caffemodel', image_dims=(56, 56))
  # Taken from:https://github.com/BVLC/caffe/issues/1774
  #net = caffe.Net('googlenet/train_val.prototxt', 'googlenet/models/googlenet_quick_iter_160000.caffemodel')
  #net.forward() # this will load the next mini-batch as defined in the net
  #fc7 = net.blobs['fc7'].data 
  
  c = 0
  fs = []
  imgs = []
  blocksize = 10
  print("-------------              Starting to make predictions   \n")
  for f in files:
    c += 1
    start = time.time()
    input_image = caffe.io.load_image(path + f)
    fs.append(f)
    imgs.append(input_image)
    if (len(fs) >= blocksize):
      s1 = time.time()
      prediction = net.predict(imgs)
      print ("Time for a single prediction " + str((time.time() - s1)))
      for i,pred in enumerate(prediction):
        res = np.exp(pred) / sum(np.exp(pred))
        s = fs[i]
        for r in res:
            s = s + ',' + "{0:.4}".format(r)
        fout.write(s + '\n')
      fs = []
      imgs = []
      print("Wrote " + str(c) + "Speed  pre sec " + str((time.time() - start) / blocksize))
  prediction = net.predict(imgs)
  for i,pred in enumerate(prediction):
    res = np.exp(pred) / sum(np.exp(pred))
    s = fs[i]
    for r in res:
        s = s + ',' + "{0:.4}".format(r)
    fout.write(s + '\n')
  print("Wrote final " + str(c) + " pre sec " + str((time.time() - start) / blocksize))
