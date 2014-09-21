import numpy
import theano
import theano.tensor as T
rng = numpy.random

N = 400
feats = 10
D = (rng.randn(N, feats), rng.randint(size=N, low=0, high=2))

# Simple Data set to be compared with R-Version
# R-Version:
#           y = c(1,1,1,1,0,0,0,0,0)
#           x = c(0.1,0.2,-0.2,0.1,0.11,1,1.1,1.2,1.3)
#         res = glm(y ~ x, family=binomial(link='logit'))
# Coefficients:
# (Intercept)            x
#        2.000       -6.733
N = 9
feats = 1
y = numpy.asarray([1,1,1,1,0,0,0,0,0])
x = numpy.asarray([0.1,0.2,-0.2,0.1,0.11,1,1.1,1.2,1.3])
x = numpy.resize(x, [9,1])
D = (x,y)

training_steps = 5000

# Declare Theano symbolic variables
x = T.matrix("x")
y = T.vector("y")
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")
print "Initial model:"
print w.get_value(), b.get_value()

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))       # Probability that target = 1
prediction = p_1
xent = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Log likelihood

#cost = xent.mean() + 0.02 * (w ** 2).sum()# Original:  The cost to minimize
cost = xent.mean()                         # The log-likelihood, to be similar to the R-Solution
gw, gb = T.grad(cost, [w, b])              # Compute the gradient of the cost
                                           # (we shall return to this in a
                                           # following section of this tutorial)

# Compile
train = theano.function(
          inputs=[x,y],
          outputs=[prediction, xent],
          updates=((w, w - 1 * gw), (b, b - 1 * gb))) #Die Learningrate war zu klein
predict = theano.function(inputs=[x], outputs=prediction)

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])

print "Final model:"
print w.get_value(), b.get_value()
print "target values for D:", D[1]
print "prediction on D:", predict(D[0])