# How do we deal with the drop of performance in batch2

The model learned in batch1 is very successfull when applied to batch1. The error on the test-set is 1 or very close to it. However, if we apply this model to the batch2 (photos taken outdoors), we find a severe decrease in the performance. We speculate that this is due to different lightening conditions in batch1 and batch2. To reduce the lightening effects we apply a LBH-transformation before learning the network.

After LBH the size of the images shrinks from 48x48 to 46x46. We change the convolution layer to do padding so that we do not have to care about the geometry too much. In a first step we check if this padding has any adversal effects on the performance on the original images and found no adversal effect.


After 40'000 iteration we found for
* batch1 logloss()0.0744725941717  acc=0.96484375
* batch2 logloss()0.616781015635  acc=0.81640625 logloss(caffe layer)=0.616781

That is still a prominent drop, but this can be further dreased by more data augmentation as shown in paper Eurographics.