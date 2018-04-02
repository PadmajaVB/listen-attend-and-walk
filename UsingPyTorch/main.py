from __future__ import unicode_literals, division
import config
import DataProcessing
import time
import models
import torch
from torch import optim
import torch.nn as nn
from torch.autograd import Variable
import random

use_cuda = torch.cuda.is_available()
teacher_forcing_ratio = 0.5
MAX_LENGTH = 25
STOP = 3


def train(idx_data, map_name, input_variable, target_variable, action_seq, encoder, decoder, encoder_optimizer,
          decoder_optimizer, criterion, processed_data, max_length=MAX_LENGTH):
    """
    Args:
        input_variable: seq_lang_numpy (torch.Tensor) -> Array
        target_variable: seq_world_numpy (torch.Tensor) -> Matrix
        action_seq: seq_action_numpy (torch.Tensor) -> Array
        encoder: Object of EncoderRNN
        decoder: Object of AttnDecoder
        encoder_optimizer: Optimizer applied to Encoder (SGD)
        decoder_optimizer: Optimizer applied to AttnDecoder (SGD)
        criterion: Loss function (NLLloss)
        processed_data: Object of ProcessData class
        max_length: Length of longest input instruction
    """
    encoder_hidden = encoder.initHidden()

    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()

    print("---------------------------", input_variable.size())
    input_length = input_variable.size()[0]  # total no. of words
    action_length = action_seq.size()[0]  # total no. action sequence

    encoder_outputs = Variable(torch.zeros(max_length, encoder.hidden_size))
    encoder_outputs = encoder_outputs.cuda() if use_cuda else encoder_outputs

    loss = 0

    """ print this  """
    count = 0
    for ei in range(input_length):
        encoder_output, encoder_hidden = encoder(input_variable[ei], encoder_hidden)
        if count == 0:
            print ("encoder_output: ", encoder_output)
            print("encoder_output.shape() : ", encoder_output.size())
            print("encoder_hidden: ", encoder_hidden)
            print("encoder_hidden.shape(): ", encoder_hidden.size())
            count = 1
        encoder_outputs[ei] = encoder_output[0][0]

    print ("In train: ", encoder_outputs)
    print()
    print()
    print ("TargetVariable[0].shape=",target_variable[0].size())
    print ("TargetVariable[0]=",target_variable[0])
    print ("TargetVariable[0]=",[target_variable[0]])

    decoder_input = target_variable[0].view(1, -1)
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden

    use_teacher_forcing = True if random.random() < teacher_forcing_ratio else False

    run_model = DataProcessing.RunModel()

    pos_start, pos_end = processed_data.get_pos(idx_data, map_name, 'train')

    pos_curr = pos_start

    if use_teacher_forcing:
        # Teacher forcing: Feed the target as the next input
        for di in range(action_length):
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            print "actionnnnnnnn",action_seq[di].data[0]
            loss += criterion(decoder_output, action_seq[di])
            if di == action_length-1 or action_seq[di].data[0] == 3:
                break
            decoder_input = target_variable[di+1]  # Teacher forcing

    else:
        # Without teacher forcing: use its own predictions as the next input
        for di in range(action_length):
            # decoder_output : (4,)
            decoder_output, decoder_hidden, decoder_attention = decoder(
                decoder_input, decoder_hidden, encoder_outputs)
            topv, topi = decoder_output.data.topk(1)
            # topv = top_value topi = top_index i.e index with highest probability
            ni = topi[0][0]  # coz, decoder_output is 3D
            '''
            'ni' is the action_sequence, so,
            TODO
            calculate the next position based on the current position
            calculate world state of next position
            decoder_input = world_state(next_position)
            '''
            # calculating next position based on highest probability
            pos_curr = run_model.take_one_step(pos_curr, ni)
            # world state of next position
            decoder_input = run_model.get_feat_current_position(pos_curr, map_name)
            decoder_input = Variable(torch.FloatTensor([decoder_input]))

            decoder_input = decoder_input.cuda() if use_cuda else decoder_input

            loss += criterion(decoder_output, action_seq[di])
            # print("ni=",ni)
            if ni == STOP:
                break

    loss.backward()

    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0] / action_length


def main():
    # TODO preprocess the input file to get standard vectors
    configuration = config.get_config()
    filepath = configuration['datafile_path']

    """divides the data into train and dev"""
    processed_data = DataProcessing.ProcessData(filepath)

    """ Model designing part """
    # TODO design encoder

    model_config = config.get_model_config()
    num_input_words = model_config['dim_lang']
    world_state_size = model_config['dim_world']
    num_output_actions = model_config['dim_action']
    hidden_size = model_config['hidden_size']
    # max_action_len = 30

    encoder = models.EncoderRNN(num_input_words, hidden_size)
    attn_decoder = models.AttnDecoderRNN (hidden_size, world_state_size, num_output_actions)
    learning_rate = model_config['learning_rate']

    """ Training part """
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

        print_loss_total = 0  # Reset every print_every
        plot_loss_total = 0  # Reset every plot_every
        encoder_optimizer = optim.SGD(encoder.parameters(), lr=learning_rate)
        decoder_optimizer = optim.SGD(attn_decoder.parameters(), lr=learning_rate)

        criterion = nn.NLLLoss()

        for name_map in configuration['maps_train']:
            max_steps = len(
                    processed_data.dict_data['train'][name_map]
            )
            print 'max_steps=', max_steps
            for idx_data, data in enumerate(processed_data.dict_data['train'][name_map]):

                # seq_lang_numpy, seq_world_numpy and seq_action_numpy will be set
                seq_lang_numpy, seq_world_numpy, seq_action_numpy = processed_data.process_one_data(idx_data, name_map,
                                                                                                    'train')

                seq_lang_numpy = Variable(torch.LongTensor(seq_lang_numpy).view(-1, 1))
                seq_world_numpy = Variable(torch.FloatTensor(seq_world_numpy))
                seq_action_numpy = Variable(torch.LongTensor(seq_action_numpy).view(-1, 1))

                """ trainer = Instantiates the model """
                print("Seq_lang: ", seq_lang_numpy)
                print("shape seq lang: ", type(seq_lang_numpy), seq_lang_numpy.size())
                print("Seq_action: ", seq_action_numpy)
                print("shape action", type(seq_action_numpy), seq_action_numpy.size())
                print("Seq_world: ", seq_world_numpy)
                print("shape world", type(seq_world_numpy), seq_world_numpy.size())

                loss = train(idx_data, name_map, seq_lang_numpy, seq_world_numpy, seq_action_numpy, encoder,
                attn_decoder, encoder_optimizer, decoder_optimizer, criterion, processed_data)

                print_loss_total += loss
                plot_loss_total += loss

                if idx_data % 100 == 99:
                    print "training i-th out of N in map : ", (idx_data, max_steps, name_map)

                if idx_data == 20:
                    break
            #
            num_steps += max_steps
        #
        train_err = err / num_steps
        break
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
