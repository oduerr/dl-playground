__author__ = 'oli'
import sys
sys.path.append('~/caffe/caffe/python/caffe')
caffe_root = '/home/dueo/caffe/caffe/' 
sys.path.insert(0, caffe_root + 'python')
import caffe
print("Finshed imports")

if __name__ == "__main__":
  if False:
    print("#############################################################")
    print("Training the model")
    print("#############################################################")
    caffe.set_mode_cpu()
    caffe.set_phase_train()
    res = caffe.SGDSolver('lenet_solver.prototxt')
    print("#############################################################")
    print("Finished Training the model")
    print("#############################################################")
    res.net.save('model/modelT')
  # Loading the classifier (no direct way?)
  net = caffe.Classifier('lenet_deploy.prototxt', 'model/lenet_iter_1000.caffemodel', image_dims=(46, 46))
  input_image = caffe.io.load_image('/home/dueo/data_kaggel_bowl/train_resized/hydromedusae_h15/138113.jpg')
  prediction = net.predict([input_image, input_image]) 
  print("The prediction is")
  print(prediction)
 
    
