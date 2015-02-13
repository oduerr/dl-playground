system("cat paper21.txt| grep 4242 > dumm.txt")
res <- read.table('dumm.txt', stringsAsFactors = FALSE, sep=",")
plot(res$V3, type = 'l',ylim=c(0,10), xlab="Epochs", ylab="Error %",  lty=1, col='black')

system("cat paper21_a.txt| grep 4242 > dumm.txt")
res <- read.table('dumm.txt', stringsAsFactors = FALSE, sep=",")
lines(res$V3, lty=2, col='red')

system("cat paper21_b.txt| grep 4242 > dumm.txt")
res <- read.table('dumm.txt', stringsAsFactors = FALSE, sep=",")
lines(res$V3, lty=2, col='red')

system("cat paper21_c.txt| grep 4242 > dumm.txt")
res <- read.table('dumm.txt', stringsAsFactors = FALSE, sep=",")
lines(res$V3, lty=1, col='green')

system("cat paper21_d.txt| grep 4242 > dumm.txt")
res <- read.table('dumm.txt', stringsAsFactors = FALSE, sep=",")
lines(res$V3, lty=2, col='green')

system("cat paper21_e.txt| grep 4242 > dumm.txt")
res <- read.table('dumm.txt', stringsAsFactors = FALSE, sep=",")
lines(res$V3, lty=2, col='green')

system("cat test_46_speck_rot_smallscaling_k20_100.txt | grep 4242 > dumm.txt")
res <- read.table('dumm.txt', stringsAsFactors = FALSE, sep=",")
lines(res$V3, lty=2, col='grey')

