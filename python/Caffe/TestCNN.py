__author__ = 'oli'
import sys
sys.path.append('~/caffe/caffe/python/caffe')
caffe_root = '/home/dueo/caffe/caffe/' 
sys.path.insert(0, caffe_root + 'python')
import caffe
print("Finshed imports")

if __name__ == "__main__":
  print("Hallo Gallo")
  caffe.set_mode_cpu()
  caffe.set_phase_train()
  #caffe.SGDSolver('lenet_train_test_files.prototxt')
  caffe.SGDSolver('')
