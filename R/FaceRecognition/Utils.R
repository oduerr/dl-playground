plotExamples <- function(y,X, title="") {
  idx = NULL
  for (i in unique(y)){
    idx <- append(idx, (which(y == i)[1]))
    #idx <- append(idx, (which(y == i)[2]))
    #idx <- append(idx, (which(y == i)[3]))
  }
  for (i in idx) {
    #rows <- ceiling(sqrt(length(idx)))
    #par(mfrow = c(rows, rows)) #does not work with image
    image(matrix(rev(X[i,]), nrow = N, ncol = N), useRaster = TRUE,axes = FALSE, col=gray((0:255)/255), main=paste0(title, " y=",y[i]))
    #par(mfrow = c(1,1))
  }
} 
