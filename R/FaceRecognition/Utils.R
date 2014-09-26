plotExamples <- function(y,X, title="", mfrow = c(4,4)) {
  par(mfrow=mfrow)
  par(mai=c(0.1,0.1,0.1,0.1))
  idx = NULL
  N = sqrt(ncol(X))
  for (i in unique(y)){
    idx.y = which(y == i)
    idx <- append(idx, idx.y[1:min(3, length(idx.y))])
    #idx <- append(idx, (which(y == i)[3]))
  }
  for (i in idx) {
    #rows <- ceiling(sqrt(length(idx)))
    #par(mfrow = c(rows, rows)) #does not work with image
    image(matrix(rev(X[i,]), nrow = N, ncol = N), useRaster = TRUE,axes = FALSE, col=gray((0:255)/255), main=paste0(title, " y=",y[i]))
    #par(mfrow = c(1,1))
  }
  par(mfrow=c(1,1))
} 
