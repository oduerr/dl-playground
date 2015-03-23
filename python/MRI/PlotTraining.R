# Comparing with random loss
#load('/home/dueo/dl-playground/R/MRI/training_testing_split.Rdata')
table(training$type)



setwd("/home//dueo/dl-playground/python/MRI/model/")
system("~/caffe/caffe/tools/extra/parse_log.sh letnet_1.txt")
train = read.table('letnet_1.txt.train', header = TRUE, comment.char = 'H')
test  =  read.table('letnet_1.txt.test', header = TRUE, comment.char = 'H')

run = data.frame(iter = c(test$X.Iters, train$X.Iters), 
           loss = c(test$TestLoss, train$TrainingLoss),
           type = c(rep('Testing', nrow(test)), rep('Training', nrow(train))))

df = run
ggplot(data=subset(df, type=='Training')) + aes(x = iter, y = loss) + geom_point(alpha=0.3, size=2) +
  geom_line(data=subset(df, type == 'Testing'), aes(x = iter, y = loss),size=2) +
  xlab('Iterations (batchsize=64)') + ylab("Log-Loss") + 
  theme(axis.title.x = element_text(colour="darkred", size=14)) + theme(axis.title.y = element_text(colour="darkred", size=14)) 
  #xlim(90000,110000) + ylim(0.5,1.5)
  #facet_grid(.~run)


 
