import numpy

def random_weights(nrow, ncol):
    bound = (numpy.sqrt(6.0) / numpy.sqrt(nrow+ncol) ) * 1.0
    # nrow -- # of prev layer units, ncol -- # of this layer units
# this is form Bengio's 2010 paper
    values = numpy.random.uniform(
        low=-bound, high=bound, size=(nrow, ncol)
    )
    return numpy.cast[float](values)

def softmax(self, x):
	# x is a vector
	exp_x = numpy.exp(x - numpy.amax(x))
	return exp_x / numpy.sum(exp_x)