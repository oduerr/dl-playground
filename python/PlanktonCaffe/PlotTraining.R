d = read.table("../sampleSubmission.csv.head.csv", sep=",", header=TRUE)
names = colnames(d)[2:122]
dd = data.frame(names = names)
row.names(dd) = 0:120
write.table(file='../labelList.csv', dd, sep="\t", quote=FALSE)

############ 
# LetNet
setwd("/home//dueo/dl-playground/python/PlanktonCaffe/lenet/")
#system("~/caffe/caffe/tools/extra/parse_log.sh log_lenet_60.txt")
train = read.table('log_lenet_60.txt.train', header = TRUE, comment.char = 'H')
plot(train[,1], train[,3], type='l', xlab='Iteration', ylab='Log-Loss', main='LeNet (no shuffeling)', ylim=c(0,5))
test = read.table('log_lenet_60.txt.test', header = TRUE, comment.char = 'H')
lines(test[,1], test[,4], col='blue')

#system("~/caffe/caffe/tools/extra/parse_log.sh log_lenet_dropout.txt")
train.do = read.table('log_lenet_dropout.txt.train', header = TRUE, comment.char = 'H')
plot(train.do[,1], train.do[,3], col='red', type='l', main='LeNet with shuffle and rotation and Dropout')
test.do = read.table('log_lenet_dropout.txt.test', header = TRUE, comment.char = 'H')
lines(test.do[,1], test.do[,4], col='green', lw = 5)
abline(h=1.40)

#system("~/caffe/caffe/tools/extra/parse_log.sh log_lenet_60_shuffel.log")
train = read.table('log_lenet_60_shuffel.log.train', header = TRUE, comment.char = 'H')
lines(train[,1], train[,3], col='red', type='l')
test = read.table('log_lenet_60_shuffel.log.test', header = TRUE, comment.char = 'H')
lines(test[,1], test[,4], col='blue', lw = 5)
abline(h=1.40)

library(ggplot2)
gg <- ggplot()
gg <- gg + geom_line(aes(x = train.do$X.Iters, y = train.do$TrainingLoss, colour='training dropout'), size=0.25) 
gg <- gg + geom_line(aes(x = test.do$X.Iters,  y = test.do$TestLoss, colour='testing dropout'), size=2)
gg <- gg + geom_line(aes(x = train$X.Iters, y = train$TrainingLoss, colour='training'), size=0.25) 
gg <- gg + geom_line(aes(x = test$X.Iters,  y = test$TestLoss, colour='testing'), size=2)
gg <- gg + geom_hline(y=1.44)
gg <- gg + ggtitle('Let-Net with or without dropout')
gg + theme_light() + xlab('Iterations') + ylab('log-loss') + scale_color_manual(values=c('red','blue','red','blue'))


################
# alex net (on 256x256)
setwd("/home//dueo/dl-playground/python/PlanktonCaffe/alexnet/")
system("~/caffe/caffe/tools/extra/parse_log.sh 27Feb.log")
train = read.table('27Feb.log.train', header = TRUE, comment.char = 'H')
test = read.table('27Feb.log.test', header = TRUE, comment.char = 'H')

library(ggplot2)
gg <- ggplot()
gg <- gg + geom_line(aes(x = train$X.Iters, y = train$TrainingLoss, colour='training'), size=0.25) 
gg <- gg + geom_line(aes(x = test$X.Iters,  y = test$TestLoss, colour='testing'), size=2) #Wrong naming
gg <- gg + geom_hline(y=1.44)
gg + geom_point(aes(x = test$X.Iters,  y = test$TestLoss, colour='testing dropout')) +theme_light() + xlab('Iterations') + ylab('log-loss')



################
# Google Net
setwd("/home//dueo/dl-playground/python/PlanktonCaffe/googlenet/")
system("~/caffe/caffe/tools/extra/parse_log.sh log_googlenet_22_feb.log")
train = read.table('log_googlenet_22_feb.log.train', header = TRUE, comment.char = 'H')
test = read.table('log_googlenet_22_feb.log.test', header = TRUE, comment.char = 'H')

library(ggplot2)
gg <- ggplot()
gg <- gg + geom_line(aes(x = train$X.Iters, y = train$TrainingLoss, colour='training dropout'), size=0.25) 
gg <- gg + geom_line(aes(x = test$X.Iters,  y = test$TestAccuracy, colour='testing dropout'), size=2) #Wrong naming
gg <- gg + geom_hline(y=1.44)
gg + geom_point(aes(x = test$X.Iters,  y = test$TestLoss, colour='testing dropout')) +theme_light() + xlab('Iterations') + ylab('log-loss')


# Image Net
#####
setwd("/home//dueo/dl-playground/python/PlanktonCaffe/imagenet/")
system("~/caffe/caffe/tools/extra/parse_log.sh imagenet_21_feb.log")
train = read.table('imagenet_21_feb.log.train', header = TRUE, comment.char = 'H')
test = read.table('imagenet_21_feb.log.test', header = TRUE, comment.char = 'H')

# Smaller images
system("~/caffe/caffe/tools/extra/parse_log.sh Alex_23_Feb.log")
train = read.table('Alex_23_Feb.log.train', header = TRUE, comment.char = 'H')
test = read.table('Alex_23_Feb.log.test', header = TRUE, comment.char = 'H')


library(ggplot2)
gg <- ggplot()
gg <- gg + geom_line(aes(x = train$X.Iters, y = train$TrainingLoss, colour='training dropout'), size=0.25) 
gg <- gg + geom_line(aes(x = test$X.Iters,  y = test$TestLoss, colour='testing dropout'), size=2)
gg + geom_point(aes(x = test$X.Iters,  y = test$TestLoss, colour='testing dropout')) +theme_light() + xlab('Iterations') + ylab('log-loss')



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

system("~/caffe/caffe/tools/extra/parse_log.sh log_caffe_60_shuffle_mean.log")
train = read.table('log_caffe_60_shuffle_mean.log.train', header = TRUE, comment.char = 'H')
plot(train[,1], train[,3], col='red', type='l')
test = read.table('log_caffe_60_shuffle_mean.log.test', header = TRUE, comment.char = 'H')
lines(test[,1], test[,4], col='brown', type='b')

