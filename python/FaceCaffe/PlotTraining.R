############ 
# LetNet
setwd("/home//dueo/dl-playground/python/FaceCaffe/model/")
system("~/caffe/caffe/tools/extra/parse_log.sh log_dumm1.txt")
train.do = read.table('log_dumm1.txt.train', header = TRUE, comment.char = 'H')
test.do = read.table('log_dumm1.txt.test', header = TRUE, comment.char = 'H')

system("~/caffe/caffe/tools/extra/parse_log.sh log_lenet.txt")
train = read.table('log_lenet.txt.train', header = TRUE, comment.char = 'H')
test = read.table('log_lenet.txt.test', header = TRUE, comment.char = 'H')


library(ggplot2)
gg <- ggplot()
gg <- gg + geom_line(aes(x = train$X.Iters, y = train$TrainingLoss, colour='training'), size=0.25) 
gg <- gg + geom_line(aes(x = test$X.Iters,  y = test$TestLoss, colour='testing'), size=2)
gg <- gg + geom_line(aes(x = train.do$X.Iters, y = train.do$TrainingLoss, colour='training.do'), size=1.25) 
gg <- gg + geom_line(aes(x = test.do$X.Iters,  y = test.do$TestLoss, colour='testing.do'), size=2) 
gg <- gg + ggtitle('Convergence on batch1 training and testset')
gg <- gg + geom_hline(y=1.79, size = 0.15)
gg + theme_light() + xlab('Iterations') + ylab('log-loss') + scale_color_manual(values=c('red','blue','red','blue'))


