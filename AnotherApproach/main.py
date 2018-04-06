import config
import DataProcessing
import time
import models
import numpy as np


def main():
	# TODO preprocess the input file to get standard vectors
	configuration = config.get_config()
	filepath = configuration['datafile_path']
	processed_data = DataProcessing.ProcessData(filepath)

	""" Model designing part """
	# TODO design encoder
	for epi in range(configuration['max_epochs']):
		#
		print "training epoch ", epi
		#
		err = 0.0
		num_steps = 0
		# TODO: shuffle the training data and train this epoch
		##
		train_start = time.time()
		#
		seq_lang_numpy = []
		seq_world_numpy = []
		seq_action_numpy = []


		for name_map in configuration['maps_train']:
			max_steps = len(
					processed_data.dict_data['train'][name_map]
			)
			print 'max_steps=', max_steps
			for idx_data, data in enumerate(processed_data.dict_data['train'][name_map]):

				# seq_lang_numpy, seq_world_numpy and seq_action_numpy will be set
				seq_lang_numpy, seq_world_numpy, seq_action_numpy = processed_data.process_one_data(idx_data, name_map, 'train')
				# np.concatenate((seq_lang_numpy, seq_lang))
				# np.concatenate((seq_world_numpy, seq_world))
				# np.concatenate((seq_action_numpy, seq_action))


				""" trainer = Instantiates the model """
				model = models.SeqToSeq()
				cost_numpy = model.build_model(
					seq_lang_numpy,  # list of word indices
					seq_world_numpy,  # matrix of dim (len(one_data['cleanpath'])*78
					seq_action_numpy  # index value of 1 in one hot vector of action
				)
				print "Cost!!------", cost_numpy
				print "type = ", type(cost_numpy)
				print "shape = ", cost_numpy.shape
				print "---Cost_numpy___=",cost_numpy
				err += cost_numpy
				if idx_data % 100 == 99:
					print "training i-th out of N in map : ", (idx_data, max_steps, name_map)
			#
			num_steps += max_steps
		#
		train_err = err / num_steps
		#
		#

# TODO design multi-level aligner

# TODO design decoder

""" Training the model """
# TODO write training code

# TODO map the o/p action sequence to the i/p instruction for effective backpropagation

""" Testing the model """
# TODO write testing code

# TODO simulate the output

if __name__ == '__main__':
	main()
