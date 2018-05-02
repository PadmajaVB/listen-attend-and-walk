import config
import DataProcessing
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 46
STOP = 3
ENCODER_PATH = "./TrainedModel/encoder.pkl"
DECODER_PATH = "./TrainedModel/decoder.pkl"


def evaluate(encoder, decoder, max_length=MAX_LENGTH):
	configuration = config.get_config()
	filepath = configuration['datafile_path']
	name_map = configuration['map_test'][0]
	processed_data = DataProcessing.ProcessData(filepath)
	run_model = DataProcessing.RunModel()
	all_actions = []
	all_attentions = []

	for idx_data, data in enumerate(processed_data.dict_data['dev'][name_map]):
		seq_lang_numpy, seq_world_numpy, seq_action_numpy = processed_data.process_one_data(idx_data, name_map, 'dev')

		seq_lang = Variable(torch.LongTensor(seq_lang_numpy).view(-1, 1))
		seq_world = Variable(torch.FloatTensor(seq_world_numpy))
		seq_action = Variable(torch.LongTensor(seq_action_numpy).view(-1, 1))

		input_length = seq_lang.size()[0]

		pos_start, pos_end = processed_data.get_pos(idx_data, name_map, 'dev')

		pos_curr = pos_start

		encoder_hidden = encoder.initHidden()

		encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
		encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

		for ei in range(input_length):
			encoder_output, encoder_hidden = encoder(seq_lang[ei], encoder_hidden)
			# encoder_outputs[ei] is an extra term when compared to that in train function
			encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

		decoder_input = seq_world[0]
		decoder_input = decoder_input.cuda() if use_cuda else decoder_input

		decoder_hidden = encoder_hidden.view(1, 1, encoder.hidden_size)

		decoded_actions = []
		decoder_attentions = torch.zeros(max_length, max_length)

		for di in range(max_length):
			decoder_output, decoder_hidden, decoder_attention = decoder(
					decoder_input, decoder_hidden, encoder_outputs)
			decoder_attentions[di] = decoder_attention.data
			topv, topi = decoder_output.data.topk(1)
			ni = topi[0][0]

			pos_curr = run_model.take_one_step(pos_curr, ni)
			# world state of next position
			decoder_input = run_model.get_feat_current_position(pos_curr, name_map)
			decoder_input = Variable(torch.FloatTensor([decoder_input]))
			decoder_input = decoder_input.cuda() if use_cuda else decoder_input

			if ni == STOP:
				decoded_actions.append(3)
				break
			else:
				decoded_actions.append(ni)

		all_actions.append(decoded_actions)
		all_attentions.append(decoder_attentions[:di + 1])

		print "decoded action = ", decoded_actions

	return all_actions, all_attentions


def SampleTest(encoder, decoder, idx_data, sentence, max_length=MAX_LENGTH):
	configuration = config.get_config()
	filepath = configuration['datafile_path']
	name_map = configuration['map_test'][0]
	processed_data = DataProcessing.ProcessData(filepath)
	run_model = DataProcessing.RunModel()
	all_actions = []
	all_attentions = []
	print "Given instruction:   ", sentence

	seq_lang_numpy, seq_world_numpy, seq_action_numpy = processed_data.process_one_data(idx_data, name_map, 'dev')

	seq_lang = Variable(torch.LongTensor(seq_lang_numpy).view(-1, 1))
	seq_world = Variable(torch.FloatTensor(seq_world_numpy))
	seq_action = Variable(torch.LongTensor(seq_action_numpy).view(-1, 1))

	input_length = seq_lang.size()[0]

	pos_start, pos_end = processed_data.get_pos(idx_data, name_map, 'dev')

	pos_curr = pos_start

	encoder_hidden = encoder.initHidden()

	encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
	encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

	for ei in range(input_length):
		encoder_output, encoder_hidden = encoder(seq_lang[ei], encoder_hidden)
		# encoder_outputs[ei] is an extra term when compared to that in train function
		encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

	decoder_input = seq_world[0]
	decoder_input = decoder_input.cuda() if use_cuda else decoder_input

	decoder_hidden = encoder_hidden.view(1, 1, encoder.hidden_size)

	decoded_actions = []
	decoder_attentions = torch.zeros(max_length, max_length)

	for di in range(max_length):
		decoder_output, decoder_hidden, decoder_attention = decoder(
				decoder_input, decoder_hidden, encoder_outputs)
		decoder_attentions[di] = decoder_attention.data
		topv, topi = decoder_output.data.topk(1)
		ni = topi[0][0]

		pos_curr = run_model.take_one_step(pos_curr, ni)
		# world state of next position
		decoder_input = run_model.get_feat_current_position(pos_curr, name_map)
		decoder_input = Variable(torch.FloatTensor([decoder_input]))
		decoder_input = decoder_input.cuda() if use_cuda else decoder_input

		if ni == STOP:
			decoded_actions.append(3)
			break
		else:
			decoded_actions.append(ni)

	print "decoded action = ", decoded_actions

	return decoded_actions, decoder_attentions


def showAttention(input_sentence, output_actions, attentions):
	# Set up figure with colorbar
	fig = plt.figure()
	ax = fig.add_subplot(111)
	cax = ax.matshow(attentions.numpy(), cmap='bone')
	fig.colorbar(cax)

	# Set up axes
	ax.set_xticklabels([''] + input_sentence.split(' '), rotation=90)
	ax.set_yticklabels([''] + output_actions)

	# Show label at every tick
	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

	plt.show()


def main():
	encoder = torch.load(ENCODER_PATH)
	attn_decoder = torch.load(DECODER_PATH)
	evaluate(encoder, attn_decoder)
	# input_sentence1 = "turn left at the gray stone carpet and move to the alley with octagon flooring on either side"
	# input_sentence2 = "take a left onto the brick patch and go all the way down until you get to where there are butterflies on the wall"
	# idx = 53; GT: [0, 1, 0, 0, 0, 0, 3]
	input_sentence1 = "take a left onto the red brick and go a ways down until you come to the section with the butterflies on the wall"
	# idx = 82; GT: [1, 0, 0, 0, 0, 3]
	output_actions, attentions = SampleTest(encoder, attn_decoder, 82, input_sentence1)
	showAttention(input_sentence1, output_actions, attentions)


if __name__ == '__main__':
	main()
