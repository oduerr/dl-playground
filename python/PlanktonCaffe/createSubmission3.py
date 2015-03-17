import sys
import csv,os
import numpy as np
import time
sys.path.append('~/caffe/caffe/python/caffe')
caffe_root = '/home/dueo/caffe/caffe/' 
sys.path.insert(0, caffe_root + 'python')
import caffe
import cv2
from skimage import io
from skimage.util import img_as_ubyte
from skimage.morphology import black_tophat, white_tophat, disk
selem = disk(1)

model = 'alexnet/models9/alexnet_128_test_dropout_iter_42000.caffemodel'
model_description = 'alexnet/train_val3.prototxt'

sampleSub    ='sampleSubmission.csv.head.csv' #We just need the header

#The Set for the competition
submissionName = 'submission_big8_102k_test_100_aug.txt'
predFile = 'test_kaggle_full.txt'

#The Test-set
# submissionName = 'test_big8_102k_test_100_aug.txt'
# predFile = 'test_full.txt'




GPUID = 1

# BatchSize set in the file train_val3.prototxt
batchSize = 10



# Creates a 99x99 array using a central crop (i=0) or a random rotation and substracts the mean
def getImage(i, imOriginal, mean):
    if i == 0: #return a central crop of 99x99 pixels
        resized = cv2.resize(imOriginal, (128, 128))
        resized = resized * 256.0 - mean
        return resized[14:113, 14:113]
    rot = np.random.uniform(0,360,1).astype(int)[0] #Random rotations
    rot = 90 * np.random.uniform(0,4,1).astype(int)[0] #Random rotations
#    rot = 0
    im_size = imOriginal.shape[0]
    if (np.random.rand() > 0.5):
       if (np.random.rand() > 0.5):
         imOriginal = cv2.flip(imOriginal,0)
       else:
         imOriginal = cv2.flip(imOriginal,1)
    scale = np.random.uniform(0.9,1.1)
    mat = cv2.getRotationMatrix2D((im_size / 2, im_size / 2), rot, scale=scale)
    resized = cv2.warpAffine(img_out, mat, (im_size, im_size), borderValue=(255,255,255))
    img_out = np.zeros((resized.shape[0], resized.shape[1], 3), dtype=np.uint8)
    img_orig = resized[:,:,0]
    img_btop = 255-black_tophat(img_orig, selem)
    img_wtop = 255-white_tophat(img_orig, selem)
    img_out[:, :, 1] = img_btop
    img_out[:, :, 2] = img_wtop
    
    resized = cv2.resize(img_out, (128, 128))
    
    resized = resized * 256.0 - mean #Geht richtig in den Keller auf Werte um 4, wenn man mean nicht abzieht
#    offsetX = np.random.uniform(10,18,1).astype(int)[0] 
#    offsetY = np.random.uniform(10,18,1).astype(int)[0] 
    offsetX = np.random.uniform(0,28,1).astype(int)[0] #Random rotations
    offsetY = np.random.uniform(0,28,1).astype(int)[0] #Random rotations
    return resized[offsetY:(99+offsetY), offsetX:(99+offsetX)]


def getCondensed(file, mean):
  inn = caffe.io.load_image(file, color=False)
  blob = np.empty((batchSize,1,99,99))
  for i in range(0,batchSize): #We create 10 distortions
    blob[i,0,:,:] = getImage(i, inn, mean)
    #getImage(0, inn, mean) gives Log-Loss : 0.891441555588 on Leaderboard 0.897698
  print(str(np.min(blob)) + " " + str(np.max(blob)))
  #inn_batch=np.asarray([[inn_resized]])
  #inn_batch = inn_batch.transpose(0,1,2,3)
  net.forward_all(data=blob)
  res = net.blobs['fc8'].data
  ep = np.exp(res)
  summe = np.sum(ep, axis=1)
  pVals = ep / summe[:, np.newaxis] #vielleicht hat absolut betrag was zu tun, es wird etwas besser Log-Loss : 0.883094597633
  #return np.maximum(pVals, axis=0) #Not not normalized but does not matter
  return np.mean(pVals, axis=0) 

if __name__ == "__main__":
  # Getting the header
  fc = csv.reader(file(sampleSub))
  fst =  fc.next()
  # Opening the submission file
  fout = open(submissionName, 'w');
  w = csv.writer(fout);
  w.writerow(fst)
  
  # Setting up the net
  caffe.set_phase_test()
  caffe.set_mode_gpu()
  caffe.set_device(GPUID) #Run on GPU 2
  print('loading model')
  net = caffe.Net(model_description, model)
  print('loaded model')


  # Loading the mean file (see https://github.com/BVLC/caffe/issues/290)
  blob = caffe.proto.caffe_pb2.BlobProto()
  data = open('/home/dueo/data_kaggel_bowl/train_augmented_mean.binaryproto', 'rb').read()
  blob.ParseFromString(data)
  mean = np.array(caffe.io.blobproto_to_array(blob))[0,0,:,:] #A 128x128 image
  
  # Looping over the validation set
  read = 0
  with open(predFile, 'rb') as f:
    reader = csv.reader(f, delimiter=' ')
    for row in reader:
      names = row[0].split("/")[-1] #This file contains 130'000 lines like: /home/dueo/data_kaggel_bowl/test//1.jpg 42
      start = time.time()
      file = row[0]
      pVals = getCondensed(file, mean)
      pVals = pVals / sum(pVals)
      time4One = (time.time() - start)
      print ("Time for getting the prediction " + str(time4One) + " " + str(read) + " " + str((130000 - read)*time4One))
      line = names
      for r in pVals[:,0,0]:
        line = line + ',' + "{0:.4}".format(r)
      fout.write(line + '\n')
      read += 1
  fout.close()
        
  





