__author__ = 'oli'
import sys
sys.path.append('~/caffe/caffe/python/caffe')

'''
Needs to be called on 
LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib
PYTHONPATH="${PYTHONPATH}:/home/dueo/caffe/caffe/python";export PYTHONPATH;printenv PYTHONPATH
PATH="${PATH}:/home/dueo/caffe/caffe/python";export PATH;printenv PATH
'''
# Alterantively
# See http://codrspace.com/Jaleyhd/caffe-python-tutorial/
caffe_root = '/home/dueo/caffe/caffe/' 
sys.path.insert(0, caffe_root + 'python')
import caffe
print("Finshed imports")

if __name__ == "__main__":
  print("Hallo Gallo")
  
  # Example of laoding a pretrained model
  caffe.set_mode_cpu()
  caffe.set_phase_train()
  MODEL_FILE = '/home/dueo/caffe/caffe/models/bvlc_reference_caffenet/deploy.prototxt'
  PRETRAINED = '/home/dueo/caffe/caffe/models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
  IMAGE_FILE = '/home/dueo/caffe/caffe/examples/images/cat.jpg'
  net = caffe.Classifier(MODEL_FILE, PRETRAINED,
                       channel_swap=(2,1,0),
                       raw_scale=255,
                       image_dims=(256, 256))
  input_image = caffe.io.load_image(IMAGE_FILE)
  prediction = net.predict([input_image])  # predict takes any number of images, and formats them for the Caffe net automatically
  print 'prediction shape:', prediction[0].shape
  print 'predicted class:', prediction[0].argmax()
  [(k, v.data.shape) for k, v in net.blobs.items()]
  
  # Example of laoding a pretrained model
  
  
