createOverview <- function(filename) {
  dcmImages <- readDICOM(filename, verbose = TRUE,recursive = FALSE, exclude = "sql")
  dcm.info <- dicomTable(dcmImages$hdr)
  
  loc = as.numeric(dcm.info$"0020-1041-SliceLocation")
  idx = sort(loc, index.return=TRUE)$ix
  
  #paste("Z-Loc=",loc[1], ", dim",paste(dim(dcmImages$img[[1]])))
  print(paste0("Filename [", filename, "]"))
  print(paste0("Image information [", dcm.info$"0008-0008-ImageType"[[1]], "]"))
  print(paste0("Dimension of the images : [", dim(dcmImages$img[[1]])[1], "x", dim(dcmImages$img[[1]])[1], "]"))
  
  numImages = dim(dcm.info)[1]
  nrows = ceiling(sqrt(numImages))
  par(mfrow = c(nrows,nrows))
  par(mar = c(0,0.1,1,0))
  #par(mfrow = c(1,1));par(mar = c(2,2,2,2))
  for (i in idx) {
    pixels =dcmImages$img[[i]] 
    fact = mean(pixels) / 1000
    pixels = pixels / fact
    mean(pixels)
    #hist(pixels)
    image(t(pixels), col = grey(0:256/256), axes = FALSE,xlab = "", ylab = "", main=round(loc[i],2))
    #image(t(pixels), col = terrain.colors(256, alpha = 1), axes = FALSE,xlab = "", ylab = "", main=round(loc[i],2))
  }
}
