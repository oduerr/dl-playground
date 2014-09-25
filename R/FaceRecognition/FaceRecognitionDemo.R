trainingFile = "../data/training_48x48_aligned_large.p_R.csv.gz"
#testFile = "/Users/oli/Proj_Large_Data/PiVision/pivision/trunk/python/pickeledStuff/testing_48x48_aligned_small.p_R.csv"
testFile = "../data/testing_48x48_aligned_large.p_R.csv.gz"

source("Utils.R")
#####
# Loading the Data
# Loading the training set
dumm <- read.table(trainingFile, sep=",", stringsAsFactors = FALSE)
ddd <- as.matrix(dumm);X_training <- ddd[,-1];y_training <- ddd[,1]
N <- sqrt(ncol(X_training))
cat("Loaded Training set ", dim(X_training), " Dimension of pixels: ", N, "x", N)
plotExamples(y_training,X_training, title = "Training ")

# Loading the test set
dumm <- read.table(testFile, sep=",", stringsAsFactors = FALSE)
ddd <- as.matrix(dumm);X_testing <- ddd[,-1];y_testing <- ddd[,1]
N <- sqrt(ncol(X_testing))
cat("Loaded Test set ", dim(X_testing), " Dimension of pixels: ", N, "x", N, " number of y ", length(y_testing))
plotExamples(y_testing,X_testing, title = "Testing ")

##### 
# Detecting the principal components and diplaying them (the spucky images)
# Eigenfaces (for illustration)
fit <- princomp(t(X_training), cor=TRUE)
res.sc <- fit$scores # the principal components
par(mfrow=c(4,4))
dim(res.sc)
for (i in 1:16) {
  m <- scale(res.sc[,i])
  sm <- matrix(rev(m), ncol=N, byrow=TRUE)
  image(t(sm), useRaster=TRUE, main=NULL, col=gray.colors(255), axes = FALSE)
}
par(mfrow=c(1,1))

#####
# PCA for dimensional reduction
# The PCA for dimensional reduction is 
# PCA on trainig data (learning the transformation)
pc.cr <- prcomp(X_training, center = FALSE)
X.train.pca <- pc.cr$x
dim(X.train.pca)
plot(X.train.pca[,1], X.train.pca[,2], col=y_training)
dim(X.train.pca)

# PCA on test data (applying the transformation)
X.test.pca <- predict(pc.cr, X_testing) #
dim(X.test.pca)

##### 
# Some simple classical Methods (Eigenfaces and Fisherfaces)
# Using what is called Eigenfaces (a simple knn in the rotated space)
# Result from you openCV-pipeline (Build 246) Eigenfaces 0.720930233
# Takeing all componets (maybe restrict to the 100 best or so)
library(class)
sum(knn(train = X.train.pca, test = X.test.pca, cl = y_training) == y_testing) / length(y_testing)  # 0.7348837

# Fisher LDA 
# Note that the space has to smaller than the number of examples, hence use PCA for dimension reduction
# Result from you openCV-pipeline (Build 246) Fisherfaces 0.893023256
library(MASS)
z <- lda(X.train.pca[,1:200], y_training)
res <- predict(z, X.test.pca[,1:200])
sum(res$class == y_testing) / length(y_testing) #0.9162791

# Running a SVM after PCA
# Result from you openCV-pipeline (Build 246) 3|4 (Simple unrolling FE) 0.874418605
require(e1071)
table(as.factor(y_training))
model <- svm(X.train.pca, as.factor(y_training), kernel='linear', cost=1)
test.svm <- predict(model, X.test.pca)
table(test.svm)
sum(test.svm == y_testing)/ length(y_testing) 





