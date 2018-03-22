import numpy as np
import utils


def generate_context_vector(input_vector, hidden_state_vector):
	irows = input_vector.shape[1]
	icols = input_vector.shape[0]  # TODO: replace this with timesteps
	input_wt = utils.random_weights(irows, icols)
	# print
	# print "weights ----------------------------------------------------"
	# print input_wt

	hrows = hidden_state_vector.shape[1]
	hcols = hidden_state_vector.shape[0]
	hidden_wt = utils.random_weights(hrows, hcols)

	beta = np.tanh(np.dot(input_vector, input_wt) + np.dot(hidden_state_vector, hidden_wt))

	alpha = utils.softmax(beta)

	context_vector = np.multiply(alpha, np.concatenate((input_vector, hidden_state_vector), axis=1))

	return context_vector
