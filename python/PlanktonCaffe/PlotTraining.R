d = read.table("../sampleSubmission.csv.head.csv", sep=",", header=TRUE)
names = colnames(d)[2:122]
dd = data.frame(names = names)
row.names(dd) = 0:120
write.table(file='../labelList.csv', dd, sep="\t", quote=FALSE)

###########
# Test resuming
setwd("/home//dueo/dl-playground/python/PlanktonCaffe/alexnet/")
#system("~/caffe/caffe/tools/extra/parse_log.sh log.txt")
train.full = read.table('bigrun.txt.train', header = TRUE, comment.char = 'H')
test.full =  read.table('bigrun.txt.test', header = TRUE, comment.char = 'H')
#system("~/caffe/caffe/tools/extra/parse_log.sh bigrun_large_scaling.txt")
#system('cat bigrun_large_scaling.txt.train | awk \'{print $1\" \"$2\" \"$3\" 0.1\"}\' > d.txt}') #Training often corrupted
train.restart = read.table('d.txt', header = TRUE, comment.char = 'H')
test.restart =  read.table('bigrun_large_scaling.txt.test', header = TRUE, comment.char = 'H')
qplot() +
  geom_point(aes(x = train.full$X.Iters, y = train.full$TrainingLoss, colour ='1st Training')) +
  geom_point(aes(x = train.restart$X.Iters, y = train.restart$TrainingLoss, colour='2nd Training')) +
  geom_line(aes(x = test.full$X.Iters,  y = test.full$TestLoss, colour='1st test'), size=2) +
  geom_line(aes(x = test.restart$X.Iters,  y = test.restart$TestLoss, colour='2nd Test'), size=2) +
  xlim(0,20000)


##################
# Big Run
setwd("/home//dueo/dl-playground/python/PlanktonCaffe/alexnet/")
system("~/caffe/caffe/tools/extra/parse_log.sh bigrun.txt")
train.full = read.table('bigrun.txt.train', header = TRUE, comment.char = 'H')
test.full =  read.table('bigrun.txt.test', header = TRUE, comment.char = 'H')
ggplot() +
  geom_line(aes(x = train.full$X.Iters, y = train.full$TrainingLoss, colour='Train')) +
  geom_line(aes(x = train.restart$X.Iters,  y = train.restart$TrainingLoss, colour='resume')) +
  geom_line(aes(x = test.full$X.Iters,  y = test.full$TestLoss, colour='Test')) + 
  xlab('Iterations') + ylab('log-loss') +
  geom_hline(y=1.44) + geom_hline(y=1.32) + geom_hline(y=1.203)



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
#system("~/caffe/caffe/tools/extra/parse_log.sh 27Feb.log")
#train.old = read.table('27Feb.log.train', header = TRUE, comment.char = 'H')
#test.old =  read.table('27Feb.log.test', header = TRUE, comment.char = 'H')

#system("~/caffe/caffe/tools/extra/parse_log.sh Log27FebMassDropOut.log")
#system("~/caffe/caffe/tools/extra/parse_log.sh 28Feb_datadropout.log")
#system("~/caffe/caffe/tools/extra/parse_log.sh 1March.log")

#system("~/caffe/caffe/tools/extra/parse_log.sh 28Feb_augmentedTrain.log")
train = read.table('28Feb_augmentedTrain.log.train', header = TRUE, comment.char = 'H')
test =  read.table('28Feb_augmentedTrain.log.test', header = TRUE, comment.char = 'H')

system("~/caffe/caffe/tools/extra/parse_log.sh 1March_nopool2_augmented.log")
trainnp = read.table('1March_nopool2_augmented.log.train', header = TRUE, comment.char = 'H')
testnp =  read.table('1March_nopool2_augmented.log.test', header = TRUE, comment.char = 'H')

#system("~/caffe/caffe/tools/extra/parse_log.sh 1March_nopool2_fc1024.log")
#trainnp1024 = read.table('1March_nopool2_fc1024.log.train', header = TRUE, comment.char = 'H')
#testnp1024 =  read.table('1March_nopool2_fc1024.log.test', header = TRUE, comment.char = 'H')
library(ggplot2)
gg <- ggplot()
gg <- gg + geom_line(aes(x = train$X.Iters, y = train$TrainingLoss, colour='training'), size=0.25) 
gg <- gg + geom_line(aes(x = test$X.Iters,  y = test$TestLoss, colour='testing'), size=2) #Wrong naming
#gg <- gg + geom_line(aes(x = train.old$X.Iters, y = train.old$TrainingLoss, colour='training_old'), size=0.25) 
#gg <- gg + geom_line(aes(x = test.old$X.Iters,  y = test.old$TestLoss, colour='testing_old'), size=2) #Wrong naming

gg <- gg + geom_line(aes(x = trainnp$X.Iters * 4, y = trainnp$TrainingLoss, colour='training_np'), size=0.25) 
gg <- gg + geom_line(aes(x = testnp$X.Iters * 4,  y = testnp$TestLoss, colour='testing_np'), size=2) #Wrong naming

#gg <- gg + geom_line(aes(x = trainnp1024$X.Iters, y = trainnp1024$TrainingLoss, colour='training_np 1024'), size=0.25) 
#gg <- gg + geom_line(aes(x = testnp1024$X.Iters,  y = testnp1024$TestLoss, colour='testing_np 1024'), size=2) #Wrong naming

gg <- gg + geom_hline(y=1.44)
gg <- gg + geom_hline(y=1.32)
gg <- gg + geom_hline(y=1.203)
gg <- gg + xlim(0,65000) + ylim(0.2,3)
gg + geom_point(aes(x = testnp$X.Iters * 4,  y = testnp$TestLoss, colour='testing dropout'), size=3) +theme_light() + xlab('Iterations') + ylab('log-loss')


plot(testnp1024$X.Iters, testnp1024$Seconds)
lines(testnp$X.Iters, testnp$Seconds)

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

