import os 
import time
import shutil
import glob


GPU = 0

##########################
# Augmentation
#python ../imageUtils/AugmentTraining.py train_full.txt /home/dueo/data_kaggel_bowl/train_augmented/
augment_cmd = 'python /home/dueo/dl-playground/python/imageUtils/AugmentTraining.py'
#augment_from = ' /home/dueo/dl-playground/python/PlanktonCaffe/train_full_100.txt'
augment_from = ' /home/dueo/dl-playground/python/PlanktonCaffe/train_full.txt'
augment_to = ' /home/dueo/data_kaggel_bowl/train_augmented/'

#####################
# List creation
#python ../imageUtils/CreateLists.py /home/dueo/data_kaggel_bowl/train_augmented/ sampleSubmission.csv.head.csv full_augmented_ 1000
list_cmd = 'python /home/dueo/dl-playground/python/imageUtils/CreateLists.py'
list_from = ' ' + augment_to
list_sample = ' /home/dueo/dl-playground/python/PlanktonCaffe/sampleSubmission.csv.head.csv'
list_name = ' /home/dueo/dl-playground/python/PlanktonCaffe/full_augmented_'

###################
# Convertion
convert_cmd = '/home/dueo/caffe/caffe/build/tools/convert_imageset'
convert_opt =  ' --resize_height=128 --resize_width=128 --gray /'
convert_list = ' ' + list_name + 'train.txt'
convert_db = '/home/dueo/data_kaggel_bowl/train_augmented_lmdb_128'

##################
# Mean calculation
# ~/caffe/caffe/build/tools/compute_image_mean /home/dueo/data_kaggel_bowl/train_augmented_lmdb_256/ /home/dueo/data_kaggel_bowl/train_augmented_mean.binaryproto
mean_cmd = '/home/dueo/caffe/caffe/build/tools/compute_image_mean ' 
mean_db = convert_db
mean_fn = '/home/dueo/data_kaggel_bowl/train_augmented_mean.binaryproto'

###################
# Running the cmd
# nohup ~/caffe/caffe/build/tools/caffe train -solver lenet_solver.prototxt --gpu=1 > log_lenet.txt
cmd_dir = '/home/dueo/caffe/caffe/build/tools/caffe'
model_dir = '/home/dueo/dl-playground/python/PlanktonCaffe/alexnet/models/'
cmd_opt = ' train -solver /home/dueo/dl-playground/python/PlanktonCaffe/alexnet/solver.auto.prototxt'

for iter in range(0,100):
  print("\n\n------ New Iteration ----")
  iter = 1
  
  aug_start = time.time()
  print("  Starting Augmentation...")
  augment_all = augment_cmd + augment_from + augment_to
  stat = os.system(augment_all)
  print("  Augmentation finished in " + str(time.time() - aug_start) + " with exitstatus " + str(stat))
  
  list_start = time.time()
  print("  Starting Creation of lists ...")
  list_all = list_cmd + list_from + list_sample + list_name + ' 1000'
  print('    ' + list_all)
  stat = os.system(list_all)
  print("  List Creation finished in " + str(time.time() - list_start) + " with exitstatus " + str(stat))
  
  convert_start = time.time()
  print("  Starting Convertion ...")
  if os.path.exists(convert_db):
    print(" Removing old DB " + convert_db)
    shutil.rmtree(convert_db)
  convert_all = convert_cmd + convert_opt + convert_list + ' ' + convert_db
  print('    ' + convert_all)
  stat = os.system(convert_all)
  print("  List Creation finished in " + str(time.time() - convert_start) + " with exitstatus " + str(stat))
  
  mean_start = time.time()
  print("  Starting creation of the mean ...")
  mean_all = mean_cmd + mean_db + ' ' + mean_fn
  print(mean_all)
  stat = os.system(mean_all)
  print("   Mean calculation Finished " + str(time.time() - mean_start) + " with exitstatus " + str(stat))
  
  res = glob.glob(model_dir + '*test*.solverstate')
  maxIter = 0
  for i,name in enumerate(res):
    iter = int(name.split('_iter_')[1].split('.')[0])
    if (iter > maxIter):
      maxIter = iter
      bestName = name
      
  caffe_start = time.time()
  print("  Starting creation traing with " + bestName)
  cmd_all = cmd_dir + cmd_opt + ' --snapshot ' + bestName
  print(cmd_all)
  stat = os.system(cmd_all)
  print("   Caffe calculation Finished " + str(time.time() - caffe_start) + " with exitstatus " + str(stat))
  



# http://caffe.berkeleyvision.org/gathered/examples/imagenet.html
# ./build/tools/caffe train --solver=models/bvlc_reference_caffenet/solver.prototxt 
#--snapshot=models/bvlc_reference_caffenet/caffenet_train_10000.solverstate
# Resuming 




