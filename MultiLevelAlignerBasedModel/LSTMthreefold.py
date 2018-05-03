from __future__ import unicode_literals, division
import config
import DataProcessing
import LSTMmodel
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import os
import datetime
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import numpy as np

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 46
STOP = 3
teacher_forcing_ratio = 0.5


def as_minutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / percent
    rs = es - s
    return '%s (- %s)' % (as_minutes(s), as_minutes(rs))


def check_position_end(pos_current, pos_destination):
    diff_pos = np.sum(
        np.abs(
            pos_current - pos_destination
        )
    )
    if diff_pos < 0.5:
        return True
    else:
        return False


def evaluate(encoder, decoder, tag_split, name_map, processed_data, run_model, max_length=MAX_LENGTH):
    all_actions = []
    all_attentions = []

    cnt_success = 0

    for idx_data, data in enumerate(processed_data.dict_data[tag_split][name_map]):
        # print "idx_data = ", idx_data, "  data = ", data['instruction']
        # print "action = ", data['action'], "  cleanpath  = ", data['cleanpath']
        # actions = []
        # for act in data['action']:
        #     actions.append(np.argmax(act))
        seq_lang_numpy, seq_world_numpy, seq_action_numpy = processed_data.process_one_data(idx_data, name_map,
                                                                                            tag_split)

        seq_lang = Variable(torch.LongTensor(seq_lang_numpy).view(-1, 1))
        seq_world = Variable(torch.FloatTensor(seq_world_numpy))
        seq_action = Variable(torch.LongTensor(seq_action_numpy).view(-1, 1))

        input_length = seq_lang.size()[0]

        pos_start, pos_end = processed_data.get_pos(idx_data, name_map, tag_split)

        pos_curr = pos_start

        encoder_hidden = encoder.initHidden()

        encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
        encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

        for ei in range(input_length):
            encoder_output, encoder_hidden = encoder(seq_lang[ei], encoder_hidden)
            # encoder_outputs[ei] is an extra term when compared to that in train function
            encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]

        world_state = seq_world[0]
        decoder_input = seq_lang
        decoder_input = decoder_input.cuda() if use_cuda else decoder_input

        decoder_h0 = encoder_hidden[0].view(1, 1, encoder.hidden_size)
        decoder_c0 = encoder_hidden[1].view(1, 1, encoder.hidden_size)
        decoder_hidden = (decoder_h0, decoder_c0)

        decoded_actions = []
        decoder_attentions = torch.zeros(max_length, max_length)

        for di in range(max_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, world_state, decoder_hidden, encoder_outputs)
            decoder_attentions[di] = decoder_attention.data
            topv, topi = decoder_output.data.topk(1)
            ni = topi[0][0]

            pos_curr = run_model.take_one_step(pos_curr, ni)
            # world state of next position
            world_state = run_model.get_feat_current_position(pos_curr, name_map)
            world_state = Variable(torch.FloatTensor(world_state))
            world_state = world_state.cuda() if use_cuda else world_state

            if ni == STOP:
                decoded_actions.append(3)
                break
            else:
                decoded_actions.append(ni)

        all_actions.append(decoded_actions)
        all_attentions.append(decoder_attentions[:di + 1])

        # print "ground truth action = ", actions
        # print "decoded action = ", decoded_actions
        if check_position_end(pos_curr, data['cleanpath'][-1]):
            cnt_success += 1
        # print "success!!"
        # if actions == decoded_actions:
        # 	print "Success!!"

        # print cnt_success

    return cnt_success, idx_data, all_actions, all_attentions


def train(idx_data, map_name, input_variable, target_variable, action_seq, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, processed_data, flag, run_model, max_length=MAX_LENGTH):
    """
    Args:
        idx_data: index of the data in pkl file
        map_name: name of the map on which the training is being done
        input_variable: seq_lang_numpy (torch.Tensor) -> Array
        target_variable: seq_world_numpy (torch.Tensor) -> Matrix
        action_seq: seq_action_numpy (torch.Tensor) -> Array
        encoder: Object of EncoderRNN
        decoder: Object of AttnDecoder
        encoder_optimizer: Optimizer applied to Encoder (SGD)
        decoder_optimizer: Optimizer applied to AttnDecoder (SGD)
        criterion: Loss function (NLLloss)
        processed_data: Object of ProcessData class
        flag: train or validate
        max_length: Length of longest input instruction
    """

    encoder_hidden = encoder.initHidden()

    if flag == "train":
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()

    input_length = input_variable.size()[0]  # total no. of words
    action_length = action_seq.size()[0]  # total no. action sequence

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    # count = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        encoder_outputs[ei] = encoder_output[0][0]

    world_state = target_variable[0]
    decoder_input = input_variable
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_h0 = encoder_hidden[0].view(1, 1, encoder.hidden_size)
    decoder_c0 = encoder_hidden[1].view(1, 1, encoder.hidden_size)
    decoder_hidden = (decoder_h0, decoder_c0)

    use_teacher_forcing = True if flag == "train" else False

    pos_start, pos_end = processed_data.get_pos(idx_data, map_name, flag)

    pos_curr = pos_start

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(action_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, world_state, decoder_hidden, encoder_outputs)
            loss += criterion(decoder_output, action_seq[di])
            if di == action_length - 1 or action_seq[di].data[0] == 3:
                break
            world_state = target_variable[di + 1]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(action_length):
            # decoder_output : (4,)
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, world_state, decoder_hidden, encoder_outputs)

            # topv = top_value topi = top_index i.e index with highest probability
            topv, topi = decoder_output.data.topk(1)

            ni = topi[0][0]  # coz, decoder_output is 3D
            '''
            'ni' is the action_sequence, so,
            TODO:
            calculate the next position based on the current position
            calculate world state of next position
            decoder_input = world_state(next_position)
            '''
            # calculating next position based on highest probability
            pos_curr = run_model.take_one_step(pos_curr, ni)
            # world state of next position
            world_state = run_model.get_feat_current_position(pos_curr, map_name)
            world_state = Variable(torch.FloatTensor(world_state))
            world_state = world_state.cuda() if use_cuda else world_state

            loss += criterion(decoder_output, action_seq[di])
            # print("ni=",ni)
            if ni == STOP:
                break

    loss.backward()

    if flag == "train":
        encoder_optimizer.step()
        decoder_optimizer.step()

    return loss.data[0] / action_length


def trainIters(encoder, attn_decoder, n_iters, learning_rate, print_every=1000, plot_every=100):
    # TODO preprocess the input file to get standard vectors
    configuration = config.get_config()
    filepath = configuration['datafile_path']

    """divides the data into train and dev"""
    processed_data = DataProcessing.ProcessData(filepath)
    run_model = DataProcessing.RunModel()

    """ Model designing part """
    # TODO design encoder
    # max_action_len = 30

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.SGD(attn_decoder.parameters(), lr=learning_rate)
    count = 0
    folds = configuration['folds']

    criterion = nn.NLLLoss()

    """ Training part """
    for epi in range(n_iters):
        #
        print "training epoch ", epi
        #
        train_err_epoch = []
        val_err_epoch = []
        accuracy_for_epoch =[]
        # TODO: shuffle the training data and train this epoch
        ##
        train_start = time.time()
        #

        for fold in range(folds):
            print "Fold: ", fold
            train_err = 0.0
            num_steps = 0
            seq_lang_numpy = []
            seq_world_numpy = []
            seq_action_numpy = []

            for name_map in configuration['maps_train'][fold]:
                max_steps = len(
                    processed_data.dict_data['train'][name_map]
                )
                print 'max_steps=', max_steps

                for idx_data, data in enumerate(processed_data.dict_data['train'][name_map]):

                    count += 1
                    # seq_lang_numpy, seq_world_numpy and seq_action_numpy will be set
                    seq_lang_numpy, seq_world_numpy, seq_action_numpy = processed_data.process_one_data(idx_data,
                                                                                                        name_map,
                                                                                                        'train')

                    seq_lang_numpy = Variable(torch.LongTensor(seq_lang_numpy).view(-1, 1))
                    seq_world_numpy = Variable(torch.FloatTensor(seq_world_numpy))
                    seq_action_numpy = Variable(torch.LongTensor(seq_action_numpy).view(-1, 1))

                    """ trainer = Instantiates the model """

                    loss = train(idx_data, name_map, seq_lang_numpy, seq_world_numpy, seq_action_numpy, encoder,
                                 attn_decoder, encoder_optimizer, decoder_optimizer, criterion, processed_data, "train",
                                 run_model)

                    train_err += loss
                    print_loss_total += loss
                    plot_loss_total += loss

                    if idx_data % 100 == 99:
                        print "training i-th out of N in map : ", (idx_data, max_steps, name_map)

                    if count % print_every == 0:
                        print_loss_avg = print_loss_total / print_every
                        print_loss_total = 0

                        print "----------------calculating training loss------------"
                        print "TimeSince=", time_since(start, count / n_iters)
                        print "Itr=", count
                        print " Percentage of code run=", count / n_iters * 100
                        print "Loss=", print_loss_avg
                        print "--------------------------------------------"
                        print ""
                        print ""

                        # if idx_data == 20:
                        #     break

                num_steps += max_steps

            #
            avg_train_err = train_err / num_steps
            train_err_epoch.append(avg_train_err)

            print "validating ... "
            #
            val_err = 0.0
            num_steps = 0
            dev_start = time.time()
            #
            for name_map in configuration['maps_train'][fold]:
                max_steps = len(processed_data.dict_data['dev'][name_map])
                for idx_data, data in enumerate(processed_data.dict_data['dev'][name_map]):
                    count += 1
                    # seq_lang_numpy, seq_world_numpy and seq_action_numpy will be set
                    seq_lang_numpy, seq_world_numpy, seq_action_numpy = processed_data.process_one_data(idx_data,
                                                                                                        name_map,
                                                                                                        'dev')

                    seq_lang_numpy = Variable(torch.LongTensor(seq_lang_numpy).view(-1, 1))
                    seq_world_numpy = Variable(torch.FloatTensor(seq_world_numpy))
                    seq_action_numpy = Variable(torch.LongTensor(seq_action_numpy).view(-1, 1))

                    """ trainer = Instantiates the model """

                    loss = train(idx_data, name_map, seq_lang_numpy, seq_world_numpy, seq_action_numpy, encoder,
                                 attn_decoder, encoder_optimizer, decoder_optimizer, criterion, processed_data, "dev",
                                 run_model)

                    val_err += loss
                    print_loss_total += loss
                    plot_loss_total += loss

                    if idx_data % 100 == 99:
                        print "training i-th out of N in map : ", (idx_data, max_steps, name_map)

                    if count % print_every == 0:
                        print_loss_avg = print_loss_total / print_every
                        print_loss_total = 0

                        print "----------------calculating validation loss------------"
                        print "TimeSince=", time_since(start, count / n_iters)
                        print "Itr=", count
                        print " Percentage of code run=", (count / n_iters) * 100
                        print "Loss=", print_loss_avg
                        print "--------------------------------------------"
                        print ""
                        print ""

                        # if idx_data == 20:
                        #     break

                num_steps += max_steps

            avg_val_error = val_err / num_steps
            val_err_epoch.append(avg_val_error)
            print "Epoch = ", epi, "  Train error = ", avg_train_err, "  Validation error = ", avg_val_error, " diff = "\
                , avg_train_err - avg_val_error

            # Add testing code here
            test_map_name = configuration['map_test'][fold]
            print "maptest = ", test_map_name
            cnt_success = 0
            tag_split = 'train'
            cnt, total_tuples1, _, _ = evaluate(encoder, attn_decoder, tag_split, test_map_name, processed_data, run_model)
            cnt_success += cnt
            tag_split = 'dev'
            cnt, total_tuples2, _, _ = evaluate(encoder, attn_decoder, tag_split, test_map_name, processed_data, run_model)
            cnt_success += cnt
            accuracy_for_fold = (cnt_success / ((total_tuples1 + total_tuples2) * 1.0)) * 100
            accuracy_for_epoch.append(accuracy_for_fold)
            print "Accuracy:for fold:", fold, "= ", accuracy_for_fold, " %"

        avg_train_err_epi = (sum(train_err_epoch)/3.0)
        avg_val_error_epi = (sum(val_err_epoch) / 3.0)
        avg_accuracy_epi = (sum(accuracy_for_epoch) / 3.0)
        print "Average train error for epoch ", epi, ": ", avg_train_err_epi
        print "Average val error for epoch ", epi, ": ", avg_val_error_epi
        print "Average accuracy for epoch ", epi, ": ", avg_accuracy_epi
        print "Train error - val error : ", avg_train_err_epi - avg_val_error_epi

        # Save the model after every epoch
        tracks = configuration['save_filepath']
        id_process = os.getpid()
        time_current = datetime.datetime.now().isoformat()
        tag_model = '_PID=' + str(id_process) + '_TIME=' + time_current
        path_track = tracks + 'track' + "_3FoldEpoch_" + str(epi) + "_" + tag_model + '/'

        command_mkdir = 'mkdir -p ' + os.path.abspath(
            path_track
        )
        os.system(command_mkdir)
        #

        ENCODER_PATH = path_track + 'encoder.pkl'
        DECODER_PATH = path_track + 'decoder.pkl'
        torch.save(encoder, ENCODER_PATH)
        torch.save(attn_decoder, DECODER_PATH)


def main():
    model_config = config.get_model_config()
    num_input_words = model_config['dim_lang']
    world_state_size = model_config['dim_world']
    num_output_actions = model_config['dim_action']
    hidden_size = model_config['hidden_size']
    learning_rate = model_config['learning_rate']

    encoder = LSTMmodel.EncoderRNN(num_input_words, hidden_size, bidirectionality=True)
    attn_decoder = LSTMmodel.AttnDecoderRNN(num_input_words, hidden_size, world_state_size, num_output_actions)

    trainIters(encoder, attn_decoder, 2, learning_rate)


if __name__ == '__main__':
    main()


# def get_data_tuple(sentence, processed_data, map_name):
# 	sentence = sentence.split()
# 	idx_data = -1
# 	data_tuple = None
# 	for idx, datatuple in enumerate(processed_data.raw_data[map_name]):
# 		if sentence == datatuple['instruction']:
# 			idx_data = idx
# 			idx_data = 53
# 			data_tuple = datatuple
# 			print "Index : ", idx_data
# 			print "Dataaaaaaaaaaaa : ", data_tuple
# 			break
# 	return idx_data, data_tuple['cleanpath']
#
#
# def SampleTest(encoder, decoder, sentence, map_name, max_length=MAX_LENGTH):
# 	configuration = config.get_config()
# 	filepath = configuration['datafile_path']
# 	name_map = configuration['map_test']
# 	processed_data = DataProcessing.ProcessData(filepath)
# 	idx_data, path = get_data_tuple(sentence, processed_data, map_name)
#
# 	run_model = DataProcessing.RunModel()
# 	all_actions = []
# 	all_attentions = []
# 	print "Given instruction:   ", sentence
#
# 	seq_lang_numpy, seq_world_numpy, seq_action_numpy = processed_data.process_one_data(idx_data, name_map, 'train')
#
# 	seq_lang = Variable(torch.LongTensor(seq_lang_numpy).view(-1, 1))
# 	seq_world = Variable(torch.FloatTensor(seq_world_numpy))
# 	seq_action = Variable(torch.LongTensor(seq_action_numpy).view(-1, 1))
#
# 	input_length = seq_lang.size()[0]
#
# 	pos_start, pos_end = processed_data.get_pos(idx_data, name_map, 'train')
#
# 	pos_curr = pos_start
#
# 	encoder_hidden = encoder.initHidden()
#
# 	encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
# 	encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs
#
# 	for ei in range(input_length):
# 		encoder_output, encoder_hidden = encoder(seq_lang[ei], encoder_hidden)
# 		# encoder_outputs[ei] is an extra term when compared to that in train function
# 		encoder_outputs[ei] = encoder_outputs[ei] + encoder_output[0][0]
#
# 	world_state = seq_world[0]
# 	decoder_input = seq_lang
# 	decoder_input = decoder_input.cuda() if use_cuda else decoder_input
#
# 	decoder_h0 = encoder_hidden[0].view(1, 1, encoder.hidden_size)
# 	decoder_c0 = encoder_hidden[1].view(1, 1, encoder.hidden_size)
# 	decoder_hidden = (decoder_h0, decoder_c0)
#
# 	decoded_actions = []
# 	decoder_attentions = torch.zeros(max_length, max_length)
#
# 	for di in range(max_length):
# 		decoder_output, decoder_hidden, decoder_attention = decoder(
# 				decoder_input, world_state, decoder_hidden, encoder_outputs)
# 		decoder_attentions[di] = decoder_attention.data
# 		topv, topi = decoder_output.data.topk(1)
# 		ni = topi[0][0]
#
# 		pos_curr = run_model.take_one_step(pos_curr, ni)
# 		# world state of next position
# 		world_state = run_model.get_feat_current_position(pos_curr, name_map)
# 		world_state = Variable(torch.FloatTensor(world_state))
# 		world_state = world_state.cuda() if use_cuda else world_state
#
# 		if ni == STOP:
# 			decoded_actions.append(3)
# 			break
# 		else:
# 			decoded_actions.append(ni)
#
# 	print "decoded action = ", decoded_actions
# 	# print "attention: ", decoder_attentions[:di + 1]
# 	# print "att2: ", decoder_attentions
#
# 	return decoded_actions, decoder_attentions[:di + 1], path
#
#
# def showAttention(input_sentence, output_actions, attentions):
# 	# Set up figure with colorbar
# 	fig = plt.figure()
# 	ax = fig.add_subplot(111)
# 	cax = ax.matshow(attentions.numpy(), cmap='bone')
# 	fig.colorbar(cax)
#
# 	# Set up axes
# 	ax.set_xticklabels([''] + input_sentence.split(' '), rotation=90)
# 	ax.set_yticklabels([''] + output_actions)
#
# 	# Show label at every tick
# 	ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
# 	ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
#
# 	plt.show()
#
#
# def main():
# 	parser = argparse.ArgumentParser(
# 		description='Testing model ... '
# 	)
# 	parser.add_argument(
# 		'-fp', '--FilePretrain', required=True,
# 		help='Path of the trained model'
# 	)
# 	args = parser.parse_args()
# 	assert (args.FilePretrain is not None)
# 	PATH = args.FilePretrain
# 	ENCODER_PATH = PATH + "/encoder.pkl"
# 	DECODER_PATH = PATH + "/decoder.pkl"
#
# 	map_name = "l"
#
# 	encoder = torch.load(ENCODER_PATH)
# 	attn_decoder = torch.load(DECODER_PATH)
#
# 	cnt_success = 0
# 	tag_split = 'train'
# 	cnt, _, _ = evaluate(encoder, attn_decoder, tag_split)
# 	cnt_success += cnt
# 	tag_split = 'dev'
# 	cnt, _, _ = evaluate(encoder, attn_decoder, tag_split)
# 	cnt_success += cnt
#
# 	print "Accuracy: ", (cnt_success/1070.0) * 100, " %"
#
# 	# input_sentence1 = "go forward about two segments to the intersection containing the hatrack"
# 	# input_sentence1 = "when the hall ends again take additional right"
# 	input_sentence1 = "take a left onto the brick patch and go the way down until you get to where there are butterflies on the wall"
#
# 	input_sentence1 = input_sentence1.lower()
# 	output_actions, attentions, path = SampleTest(encoder, attn_decoder, input_sentence1, map_name)
# 	# showAttention(input_sentence1, output_actions, attentions)
# 	print "Path = ", path
#
#
# if __name__ == '__main__':
# 	main()
