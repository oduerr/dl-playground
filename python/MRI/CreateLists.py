import os
import sys
import subprocess
from __builtin__ import len
import random
import cv2
import numpy as np
import theano
import theano.tensor as T
import csv

if __name__ == '__main__':
  #path = '/home/dueo/data/inselpng/training/'
  path = '/home/dueo/data/inselpng/testing/'
  import os
  classes = [os.path.join(dp, f) for dp, dn, fn in os.walk(path) for f in fn]
  try:
    classes.remove('.DS_Store')
  except:
    pass
  fout = open('testing.txt', 'w');
  random.shuffle(classes)
  for line in classes:
    type = 1
    if (line.find('gbm') > 0):
      type = 0
    fout.write(line + '\t' + str(type) + '\n')
  fout.close();
  


