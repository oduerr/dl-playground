__author__ = 'oli'
# See also: http://deeplearning.net/software/theano/tutorial/adding.html

import theano.tensor as T
from theano import function

######################################
# Addition von 2 Skalaren
x = T.dscalar('x') # x ist eine *symbolisches* Objekt a.k.a. Variable
y = T.dscalar('y') # Hier ein Skalar vom Typ double
print(type(y)) #Technisch eine Instanz von theano.tensor.var.TensorVariable

symbol = x + y**2
print(type(symbol)) #theano.tensor.var.TensorVariable
f = function([x, y], symbol) #Hier wird C-Code produziert, compiliert

print(f(2,3))
print(f(2,3.3))
print(type(f(3,3))) #numpy.ndarray

###############
# Addition von einer Matrix
x = T.dmatrix('x')
#...

if 0 > 10:
    #######################
    # Logistische Funktion, wird Elementweise ausgefuehrt
    x = T.dmatrix('x')
    s = 1 / (1 + T.exp(-x))
    logistic = function([x], s) #Immer []=list auch wenn nur ein Element in der Liste ist

    print(logistic([[0,1],[-1,2]]))

    ########################
    # Mehr als eine Berechnung in einer Funktion
    # Wahrscheinlich ist es einfacher mehr als eine Berechnung in einer Funktion zu machen
    # wenn die Daten schon daliegen

    logisticAndSquare = function([x], [s, x**2])
    print(logisticAndSquare([[0,1],[-1,2]])) #Gibt auch noch das Quadrat zurueck

#################################
# Shared Variables having state
if (False):
    from theano import shared
    state = shared(0) #Ist eine symbolische Variable deren Wert zwischen verschiedenen Funktionen
                      #Geteilt wird.
    inc = T.iscalar('inc')
    # updates = werden bei jedem Funktionsaufruf ausgefuehrt und: updates
    # must be supplied with a list of pairs of the form (shared-variable, new expression).
    accumulator = function([inc], state, updates=[(state, state+inc)])
    for i in range(0, 10):
        if (i == 5):
            state.set_value(0)
        print(accumulator(1)) # 0,1...,4,0,1,..,4

#################################
# Noch eine Besonderheit mit givens (TODO, oder auch egal)

#################################
# Random Variables in Theano
from theano.tensor.shared_randomstreams import RandomStreams
from theano import function
from theano import pp
srng = RandomStreams(seed=234)
rv_u = srng.uniform((2,2)) #Ein stream von 2x2
rv_n = srng.normal((2,2))
f = function([], rv_u)
g = function([], rv_n, no_default_updates=True)     #Not updating rv_n.rng
#In diesem Ausdruck wird rv_u nur einmal gezogen, es kommt also immer die 0 raus.
nearly_zeros = function([], rv_u + rv_u - 2 * rv_u)
print(f())
print(f())
print(g())
print(g()) #Gleichen Zahlen kommen nochmal
print(nearly_zeros()) #=0 da siehe oben

# Seeding Streams
# Will man nur den Startwert des rv_u Streams setzen, so kann man dies wie folgt tun
rng_val = rv_u.rng.get_value(borrow=True)   # Get the rng for rv_u
rng_val.seed(89234)                         # seeds the generator
rv_u.rng.set_value(rng_val, borrow=True)    # Assign back seeded rng

# Will man alle Setzen



