# See also: http://www.r-bloggers.com/things-to-try-after-user-part-1-deep-learning-with-h2o/ for a good explanation


library(h2o)

## Start a local cluster with 1GB RAM (default)
localH2O = h2o.init()
# Look at the webbrowser for additionl info on your cluster
# ??h2o help page

# Manual splitting in a trainig and test-set (validation set)
train.idx = sample(x = 1:150, size = 120, replace = FALSE)
train = iris[train.idx, ]
test = iris[-train.idx, ]

# uploading the data
train_h2o <- as.h2o(localH2O, train, key = 'dat_train')
test_h2o <- as.h2o(localH2O, test, key='dat_test') #Achtung: key must be unique
# See http://127.0.0.1:54321/StoreView.html for the uploaded data

model <- 
  h2o.deeplearning(x = 1:4,  # column numbers for predictors
                   y = 5,   # column number for label
                   data = train_h2o, # data in H2O format
                   activation = "Tanh", 
                   balance_classes = TRUE, 
                   hidden = c(50,50,50), # three layers of 50 nodes
                   epochs = 100) # max. no. of epochs

h2o_yhat_test <- h2o.predict(model, test_h2o)

## Converting H2O format into a data frame
df_yhat_test <- as.data.frame(h2o_yhat_test)
df_yhat_test$True <- test$Species

# Checking the crossvalidation error on a testset
sum(df_yhat_test$predict == df_yhat_test$True) / dim(df_yhat_test)[1] #0.9666

h2o.shutdown(localH2O, prompt = FALSE)
