par(mfrow = c(1,1))
par(mar=c(5.1,4.1,4.1,2.1))
#system("cat withperm.txt | grep 4242 > dumm.txt")
#system("cat bigrot.txt | grep 4242 > dumm.txt")
system("cat bigkernel.txt | grep 4242 > dumm.txt")
res <- read.table('dumm.txt', stringsAsFactors = FALSE, sep=",")
plot(res$V5, type = 'l',ylim=c(0,5), xlab="Epochs", ylab="Error %",  lty=2, col='blue')
lines(1:length(res$V6), res$V6, col='green', lty=1)
lines(1:length(res$V7), res$V7, col='red', lty=1)
legend("topright", legend = c("Log-Loss (Training) ", "Log-Loss (Validation) ", "Log-Loss (Testing)"), lty=c(2,1,1), col=c('blue','green','red'))
abline(h=1.35)
abline(h=1.71)





