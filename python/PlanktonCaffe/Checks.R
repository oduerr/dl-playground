r <- read.delim('test.txt', sep=" ")
d = table(r[,2])
min(d)
