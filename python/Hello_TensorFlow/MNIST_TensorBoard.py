import input_data
import tensorflow as tf

if __name__ == '__main__':
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

    # Variables entering the graph
    x = tf.placeholder("float", shape=[None, 28*28]) #Batchsize x Number of Pixels
    y_ = tf.placeholder("float", shape=[None, 10]) #Batchsize x 10 (one hot encoded)

    # Model Variables (weights of the network)
    W = tf.Variable(tf.zeros([784,10]))
    b = tf.Variable(tf.zeros([10]))

    # Model
    p_y = tf.nn.softmax(tf.matmul(x,W) + b) #y (y_hat) has the dimension batchSize x 10

    # Performance of the model for training
    cross_entropy = -tf.reduce_sum(y_*tf.log(p_y))
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(p_y,1), tf.argmax(y_,1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(1000):
            batch = mnist.train.next_batch(50)
            sess.run(train_step, feed_dict={x: batch[0], y_: batch[1]})
            if i % 100 == 0:
                res = sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})
                print("Accuracy {}".format(res))


