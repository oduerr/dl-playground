############ 
# LetNet
setwd("/home//dueo/dl-playground/python/PlanktonCaffe/lenet/")
#system("~/caffe/caffe/tools/extra/parse_log.sh log_lenet_60.txt")
train = read.table('log_lenet_60.txt.train', header = TRUE, comment.char = 'H')
plot(train[,1], train[,3], type='l', col='blue', xlab='Iteration', ylab='Log-Loss', main='LeNet', ylim=c(0,5))
test = read.table('log_lenet_60.txt.test', header = TRUE, comment.char = 'H')
lines(test[,1], test[,4], col='green')

#system("~/caffe/caffe/tools/extra/parse_log.sh log_lenet_60_shuffel.log")
train = read.table('log_lenet_60_shuffel.log.train', header = TRUE, comment.char = 'H')
plot(train[,1], train[,3], col='red', type='l', main='LeNet with shuffle and rotation')
test = read.table('log_lenet_60_shuffel.log.test', header = TRUE, comment.char = 'H')
lines(test[,1], test[,4], col='green', lw = 5)



############ 
# Caffe Net
setwd("/home//dueo/dl-playground/python/PlanktonCaffe/caffenet/")
####
# Unshuffeled
system("~/caffe/caffe/tools/extra/parse_log.sh log_caffe_60.txt")
train = read.table('log_caffe_60.txt.train', header = TRUE, comment.char = 'H')
plot(train[,1], train[,3], type='l', col='blue', xlab='Iteration', ylab='Log-Loss', main='Caffe-Net')
test = read.table('log_caffe_60.txt.test', header = TRUE, comment.char = 'H')
lines(test[,1], test[,4], col='green')

system("~/caffe/caffe/tools/extra/parse_log.sh log_caffe_60_shuffle_lr0.1.log")
train = read.table('log_caffe_60_shuffle_lr0.1.log.train', header = TRUE, comment.char = 'H')
plot(train[,1], train[,3], col='red', type='l')
test = read.table('log_caffe_60_shuffle.log.test', header = TRUE, comment.char = 'H')
lines(test[,1], test[,4], col='brown', type='b')

