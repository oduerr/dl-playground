#test_46_lph_new_lr0.1_speckel.txt
#test_46_lph_new_lr0.01_speckel.txt
#test_46_lph_new_lr0.1_speckel50.txt
#test_46_lph_new_lr0.01.txt
#test_46_lph_new_lr0.001
#test_46_lph_new_lr0.05_speckel.txt
#test_46_lph_new_K20_lr0.1_speckel.txt
#test_46_speck_rot_scaling_k100.txt
#test_46_speck_rot_scaling_k40_200.txt
#test_46_speck_rot_smallscaling_k20_100.txt
#test_46_speck0.05_rot_smallscaling_k20_100.txt
#state_lbh_elip_scale_K100_roling.txt
#state_lbh_elip_scale_K100_roling_less.txt
#state_lbh_elip_scale_K100_no_roling_less.txt
#state_lbh_elip_scale_K100_no_roling_less_small_dist.txt

#system("cat test_46_speck_rot_smallscaling_k20_100.txt | grep 4242 > dumm.txt")
system("cat  paper21.txt | grep 4242 > paper_termalization.txt")
res <- read.table('dumm.txt', stringsAsFactors = FALSE, sep=",")
plot(res$V2, type = 'l',ylim=c(0,10), xlab="Epochs", ylab="Error %",  lty=2, col='blue')
lines(1:length(res$V3), res$V3, col='green', lty=1)
lines(1:length(res$V3), res$V5 * 100 , col='red', lty=1)
#abline(h=2.79)
(lt = mean(res$V3[800:1000]))
sd(res$V3[800:1000])
abline(h=lt)
abline(h=mean(100 * res$V5[800:1000]), lty=2)
legend("topright", legend = c("Error Validation Data (batch 1)", "Error Test Data (batch 2)", "Loglikelihood Train Data * 100"), lty=c(2,1,1), col=c('blue','green','red'))



#system("cat test_46_speck_rot_smallscaling_k20_100.txt | grep 4242 > dumm.txt")
#system("cat paper15.txt | grep 4242 > dumm.txt")
system("cat paper21_a.txt | grep 4242 > dumm.txt")
res <- read.table('dumm.txt', stringsAsFactors = FALSE, sep=",")
#plot(res$V2, type = 'l',ylim=c(0,30), xlab="Epochs", ylab="Error %",  lty=2, col='blue')
lines(res$V2, type = 'l',  lty=2, col='blue')
lines(1:length(res$V3), res$V3, col='green', lty=2)
lines(1:length(res$V3), res$V5 * 33 , col='red', lty=2)
#abline(h=2.79)
abline(h=2.31)




