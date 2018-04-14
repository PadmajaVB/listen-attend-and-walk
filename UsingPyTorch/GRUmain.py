from __future__ import unicode_literals, division
import config
import DataProcessing
import GRUmodel
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import time
import math
import os
import datetime

use_cuda = torch.cuda.is_available()
teacher_forcing_ratio = 0.5
MAX_LENGTH = 46
STOP = 3


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


def train(idx_data, map_name, input_variable, target_variable, action_seq, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, processed_data, flag, max_length=MAX_LENGTH):
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
        # if count == 0:
        #     print ("encoder_output: ", encoder_output)
        #     print("encoder_output.shape() : ", encoder_output.size())
        #     print("encoder_hidden: ", encoder_hidden)
        #     print("encoder_hidden.shape(): ", encoder_hidden.size())
        #     count = 1
        encoder_outputs[ei] = encoder_output[0][0]

    world_state = target_variable[0]
    # print("World state initially : ", world_state)
    decoder_input = input_variable
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden.view(1, 1, encoder.hidden_size)

    use_teacher_forcing = True if flag == "train" else False

    run_model = DataProcessing.RunModel()

    pos_start, pos_end = processed_data.get_pos(idx_data, map_name, 'train')

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

    """ Model designing part """
    # TODO design encoder
    # max_action_len = 30

    start = time.time()
    plot_losses = []
    print_loss_total = 0  # Reset every print_every
    plot_loss_total = 0  # Reset every plot_every
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.Adam(attn_decoder.parameters(), lr=learning_rate)
    count = 0

    criterion = nn.NLLLoss()

    """ Training part """
    for epi in range(n_iters):
        #
        print "training epoch ", epi
        #
        train_err = 0.0
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

                count += 1
                # seq_lang_numpy, seq_world_numpy and seq_action_numpy will be set
                seq_lang_numpy, seq_world_numpy, seq_action_numpy = processed_data.process_one_data(idx_data, name_map,
                                                                                                    'train')

                seq_lang_numpy = Variable(torch.LongTensor(seq_lang_numpy).view(-1, 1))
                seq_world_numpy = Variable(torch.FloatTensor(seq_world_numpy))
                seq_action_numpy = Variable(torch.LongTensor(seq_action_numpy).view(-1, 1))

                """ trainer = Instantiates the model """

                loss = train(idx_data, name_map, seq_lang_numpy, seq_world_numpy, seq_action_numpy, encoder,
                             attn_decoder, encoder_optimizer, decoder_optimizer, criterion, processed_data, flag = "train")

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

        print "validating ... "
        #
        val_err = 0.0
        num_steps = 0
        dev_start = time.time()
        #
        for name_map in configuration['maps_train']:
            max_steps = len(processed_data.dict_data['dev'][name_map])
            for idx_data, data in enumerate(processed_data.dict_data['dev'][name_map]):
                count += 1
                # seq_lang_numpy, seq_world_numpy and seq_action_numpy will be set
                seq_lang_numpy, seq_world_numpy, seq_action_numpy = processed_data.process_one_data(idx_data, name_map,
                                                                                                    'dev')

                seq_lang_numpy = Variable(torch.LongTensor(seq_lang_numpy).view(-1, 1))
                seq_world_numpy = Variable(torch.FloatTensor(seq_world_numpy))
                seq_action_numpy = Variable(torch.LongTensor(seq_action_numpy).view(-1, 1))

                """ trainer = Instantiates the model """

                loss = train(idx_data, name_map, seq_lang_numpy, seq_world_numpy, seq_action_numpy, encoder,
                attn_decoder, encoder_optimizer, decoder_optimizer, criterion, processed_data, flag="validate")

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

        avg_val_err = val_err / num_steps

        print "Epoch = ", epi, "  Train error = ", avg_train_err, "  Validation error =", avg_val_err


def main():
    model_config = config.get_model_config()
    num_input_words = model_config['dim_lang']
    world_state_size = model_config['dim_world']
    num_output_actions = model_config['dim_action']
    hidden_size = model_config['hidden_size']
    learning_rate = model_config['learning_rate']

    encoder = GRUmodel.EncoderRNN(num_input_words, hidden_size, bidirectionality=True)
    attn_decoder = GRUmodel.AttnDecoderRNN(num_input_words, hidden_size, world_state_size, num_output_actions)

    trainIters(encoder, attn_decoder, 50, learning_rate)

    id_process = os.getpid()
    time_current = datetime.datetime.now().isoformat()
    tag_model = '_PID=' + str(id_process) + '_TIME=' + time_current
    path_track = './tracks/track' + tag_model + '/'

    command_mkdir = 'mkdir -p ' + os.path.abspath(
        path_track
    )
    os.system(command_mkdir)
    #

    ENCODER_PATH = path_track + 'encoder.pkl'
    DECODER_PATH = path_track + 'decoder.pkl'
    torch.save(encoder, ENCODER_PATH)
    torch.save(attn_decoder, DECODER_PATH)


if __name__ == '__main__':
    main()
