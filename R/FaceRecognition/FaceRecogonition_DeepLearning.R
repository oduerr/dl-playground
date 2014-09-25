##############################
# Some tests using deep learing for face recognition
# Running a 'deep-neural-net' using the H20 library
# Actually a LeCun like network would be more appropriate 
library(h2o)
localH2O = h2o.init()
# Take the 
df.train = data.frame(y = as.numeric(y_training), x = X.train.pca)
train_h2o <- as.h2o(localH2O, df.train, key = 'dat_train')
test_h2o <- as.h2o(localH2O, X.test.pca, key='dat_test') #Achtung: key must be unique

model <- 
  h2o.deeplearning(x = 2:(ncol(df.train)),  # column numbers for predictors
                   y = 1,   # column number for label
                   data = train_h2o, # data in H2O format
                   activation = "Tanh", 
                   l2 = 0.2,
                   balance_classes = TRUE, 
                   hidden = c(50,50), # three layers of 50 nodes
                   epochs = 100) # max. no. of epochs

model
h2o_yhat_test <- h2o.predict(model, test_h2o)
df_yhat_test <- as.data.frame(h2o_yhat_test)
df_yhat_test$predict
h2o.shutdown(localH2O, prompt = FALSE)