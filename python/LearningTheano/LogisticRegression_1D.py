import numpy
import theano
import theano.tensor as T
rng = numpy.random

# Ein einfaches Theano Beispiel fuer logistische Regression.
# Siehe auch: James Bergstra..., Bengio PROC. OF THE 9th PYTHON IN SCIENCE CONF. (SCIPY 2010)
#   http://conference.scipy.org/proceedings/scipy2010/pdfs/proceedings.pdf

## Kurze mathematische Vorbemerkung
# Wir betrachten die Likelihood, dass eine 1 kommt gegeben den Daten x (x ist moeglicherweise ein Vektor).
#       p(y=1 | x) = exp(b + W' x) / (1 + exp(b + W' x)) = [1 + exp-(b + W' x)]^-1
# Die Likelihood fuer eine 0 ist dann einfach die Gegenwahrscheinlichkeit
# Somit ist die Likelihood fuer alle Beispiele (wir nehmen natuerlich iid an)
#       L = \prod_{y_i=1}  p(y=1 | x) * \prod_{y_i=0}  p(y=0 | x)
# Wir maximieren die Log-Likelihood l, die wir mit kompakt als
#      l = \sum y_i * log(p(y=1 | x_i)) + (1 - y_i) * log(1 - p(y=1 | x))
# schreiben.

# Fuer unsere R-Freunde
# Simple Data set to be compared with R-Version
# R-Version:
#           y = c(1,1,1,1,0,0,0,0,0)
#           x = c(0.1,0.2,-0.2,0.1,0.11,1,1.1,1.2,1.3)
#         res = glm(y ~ x, family=binomial(link='logit'))
# Coefficients:
# (Intercept)            x
#        2.000       -6.733

feats = 1
# Symbolic Variables (die braucht man um den expression graph aufzubauen)
x = T.matrix("x")
y = T.vector("y")
# Shared Variables (symbolische die Werte zwischen verschiedenen Aufrufen beibehalten)
w = theano.shared(rng.randn(feats), name="w")
b = theano.shared(0., name="b")
print "Initial model:"
print w.get_value(), b.get_value()

# Construct Theano expression graph
p_1 = 1 / (1 + T.exp(-T.dot(x, w) - b))       # p_1 = p(y=1 | x)
like = -y * T.log(p_1) - (1-y) * T.log(1-p_1) # Log likelihood

#cost = xent.mean() + 0.02 * (w ** 2).sum()# Original:  The cost to minimize
# Wir berechnen die Likelihood als Mittelwert (somit haengen die Parameter nicht so stark vom der Anzahl Trainingsbeispiele ab)
cost = like.mean()

# Hier ist die eigentliche Staerke von Theano: Wir berechnen den Gradienten.
gw, gb = T.grad(cost, [w, b])

# Compile
train = theano.function(
          inputs=[x,y],         #Geht in die Funktion rein.
          outputs=[p_1, like],  #Kommt aus der Funktion raus (koennen wir allerdings nicht als neuen Input nehmen)
          updates=((w, w - 1 * gw), (b, b - 1 * gb)) #Das wird dann als neuer input genommen
)

# Notiz ueber den update Mechanismus werden w und b uber der Gradienten gw, gb neue Werte zugewiesen.
# Das ist wahrscheinlich auf einer GPU viel schneller zu erreichen als wenn man es immer zurueckgeben muss.
# Die Learningrate war zu klein (mit 0.01)

predict = theano.function(inputs=[x], outputs=p_1)

# Jetzt verwenden wir die comiplierten Funktionen
N = 9
y = numpy.asarray([1,1,1,1,0,0,0,0,0])
x = numpy.asarray([0.1,0.2,-0.2,0.1,0.11,1,1.1,1.2,1.3])
x = numpy.resize(x, [9,1])
D = (x,y)

training_steps = 5000

# Train
for i in range(training_steps):
    pred, err = train(D[0], D[1])
    print(D[0])

print "Final model:"
print w.get_value(), b.get_value()
print "target values for D:", D[1]
print "prediction on D:", predict(D[0])