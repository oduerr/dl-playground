##############################
# Some tests using deep learing for face recognition
# Running a 'deep-neural-net' using the H20 library
# Actually a LeCun like network would be more appropriate 
library(h2o)
localH2O = h2o.init()
# Take the 

X.train = scale(X.train.pca[,1:200])
X.test.all  = scale(X.test.pca[,1:200])
idx.validation = sample(nrow(X.test.all), size = 0.3 * nrow(X.test.all))
X.test      = scale(X.test.all[setdiff(1:nrow(X.test.all), idx.validation), 1:200])
X.validation      = scale(X.test.all[idx.validation,1:200])

df.train = data.frame(y = as.numeric(y_training), x = X.train)
train_h2o <- as.h2o(localH2O, df.train, key = 'dat_train')
test_h2o <- as.h2o(localH2O, X.test, key='dat_test') #Achtung: key must be unique
df.valid = data.frame(y = as.numeric(y_testing[idx.validation]), x = X.validation)
validation_h2o <- as.h2o(localH2O, df.valid, key='dat_validation') #Achtung: key must be unique

model <- 
  h2o.deeplearning(x = 2:(ncol(df.train)),  # column numbers for predictors
                   y = 1,   # column number for label
                   data = train_h2o, # data in H2O format
                   validation = validation_h2o,
                   activation = "Tanh", 
                   l2 = 0.07,
                   quiet_mode = FALSE,
                   shuffle_training_data = TRUE,
                   hidden = c(50,50), # three layers of 50 nodes
                   epochs = 5000) # max. no. of epochs
model


h2o_yhat_test <- h2o.predict(model, test_h2o)
df_yhat_test <- as.data.frame(h2o_yhat_test)
df_yhat_test$predict
h2o.shutdown(localH2O, prompt = FALSE)
