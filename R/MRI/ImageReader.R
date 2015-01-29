#See dicom-images https://mattstats.wordpress.com/2013/05/22/dicom/
library(oro.dicom)
setwd("~/data/Inselspital_2015_01_21/")
#dcmImages <- readDICOM("GBM/AAYFX7JFVOJJ3KCZBKBFETVJTY======//AAYFX7JFVOJJ3KCZBKBFETVJTY======-20140521-0/11-DTI_R12_ADC/", verbose = TRUE,recursive = FALSE, exclude = "sql")
dcmImages <- readDICOM("GBM/L3B4QOBEF55NOH2R2VV2LMBAGI======//L3B4QOBEF55NOH2R2VV2LMBAGI======-20140522-0/12-DTI_R12_ep2d_diff_mddw_p2_ADC/", verbose = TRUE,recursive = FALSE, exclude = "sql")
dcm.info <- dicomTable(dcmImages$hdr)
#dcm.info$"0008-0008-ImageType"
#dcm.info$"0008-103E-SeriesDescription"
#dcm.info$"0040-0275-FFFE-E000-Item"


# find . -name *3scan_p3_m128_ADC* 
# metastasis
# NOM4T5FUBBKUCIUDEQT6KT3YMU
# SHBXWFZYWGR3SHGOLRXQTMUDNI
# X2VIFLF735FLVVYJJEPLAIZXUM
# GBM
# SBHPTILVYRLRG4ORCRIPZNN7TI
# IIYRAXCOXUDCVAFFZ3HRUCTGYI
# 4SMQCSO5HL4K55CNVE3X7XBGQY
# LD7ODTIUJACMDQ33ZKLHAA4NGI
# 7GNXCKL357OHGKZUJ7XUX3SD2Q
# TZDWUXYTJEGJZO5ELDMKN3Z7WE
# 6D7XIAAJ5WLH5LXBXQYL7A3GVU
# L3B4QOBEF55NOH2R2VV2LMBAGI
# YQDWPJNQGWFNCLGJMNTIMYJCRE
# PCGK2XRLH3EUN2JY6QHRW53NIQ
# 6DIA73RZVP5RNLXBXQYL7A3GVU
# EJFYV6QNMXNCYEKGCUQP7V55UY
# NUGVHSHIO2QJPGS4RK3AMITSBY
# YGMKBD6SU2NJNTYNTFJEA2OTQM
# AL4V5D55YCQ2CQJ5GLQZH2T5EA
# OQWVSS6OOEHSJVRECUOJ34KZKU
# 5GGUXDSD56WN35Y72X3WNLLS6Y
# JQAOQBXOVWCHQO5PCOJQIUFLNA
# ZS4VC2AHI7DAMJ2K6HWUW5CFEE

loc = as.numeric(dcm.info$"0020-1041-SliceLocation")
idx = sort(loc, index.return=TRUE)$ix

#paste("Z-Loc=",loc[1], ", dim",paste(dim(dcmImages$img[[1]])))
cat("Image information [", dcm.info$"0008-0008-ImageType"[[1]], "]")
cat("Dimension of the images : [", dim(dcmImages$img[[1]]), "]")

numImages = dim(dcm.info)[1]
nrows = ceiling(sqrt(numImages))
par(mfrow = c(nrows,nrows))
par(mar = c(0,0.1,1,0))
for (i in idx) {
  pixels = dcmImages$img[[i]]
  fact = mean(pixels) / 1000
  pixels = pixels / fact
  image(t(pixels), col = grey(0:3445/3445), axes = FALSE,xlab = "", ylab = "", main=round(loc[i],2))
}


