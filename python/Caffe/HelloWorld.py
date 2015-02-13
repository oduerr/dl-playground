__author__ = 'oli'
import sys
sys.path.append('~/caffe/caffe/python/caffe')

'''
Needs to be called on 
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
PYTHONPATH="${PYTHONPATH}:/home/dueo/caffe/caffe/python";export PYTHONPATH;printenv PYTHONPATH
PATH="${PATH}:/home/dueo/caffe/caffe/python";export PATH;printenv PATH
'''

import caffe

if __name__ == "__main__":
  print("Hallo Gallo")
