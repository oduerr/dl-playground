# Creates matlab code which can be copied and pasted into matlab for creating hdf5 data files out of mha files 

hdf5File = '/Users/oli/Proj_Large_Data/Deep_Learning_MRI/BRATS-2/test.h5'
startdir = '/Users/oli/Proj_Large_Data/Deep_Learning_MRI/BRATS-2/Image_Data/HG/'
files = dir(startdir, "mha",recursive=T)

for (i in 1:length(files)) {
  file = files[[i]]
  sp = strsplit(file, "/")
  name = sp[[1]][1]
  type = sp[[1]][2]
  cat(noquote(paste0("code = ReadData3D(\'", startdir, file,"\');\n")))
  if (i == 1) {
    mode = "'WriteMode', 'overwrite');\n"
  } else {
    mode = "'WriteMode', 'append');\n"
  }
  cat(noquote(paste0('hdf5write(\'/Users/oli/Proj_Large_Data/Deep_Learning_MRI/BRATS-2/test.h5\',',"\'", name , '/', type, "\', code,", mode )))
}

