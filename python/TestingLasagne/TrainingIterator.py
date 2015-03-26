__author__ = 'oli'

from SimpleNet import net1,X,y #We load the data from the last example

from nolearn.lasagne import BatchIterator


class SimpleBatchIterator(BatchIterator):

    def transform(self, Xb, yb):
        # The 'incomming' and outcomming shape is (10, 1, 28, 28)
        Xb, yb = super(SimpleBatchIterator, self).transform(Xb, yb)
        return Xb[:,:,::-1,:], yb #<--- Here we do the flipping

# Setting the new batch iterator
net1.batch_iterator_train = SimpleBatchIterator(batch_size=10)
net1.fit(X[0:1000,:,:,:],y[0:1000])