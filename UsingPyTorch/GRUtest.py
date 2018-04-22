import config
import DataProcessing
import torch
from torch.autograd import Variable
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import argparse
import numpy as np

use_cuda = torch.cuda.is_available()
MAX_LENGTH = 46
STOP = 3


def get_actions(one_hot_actions):
    actions = []
    for action in one_hot_actions:
        x = np.argmax(action)
        actions.append(x)
    return actions


def evaluate(encoder, decoder, tag_split, max_length=MAX_LENGTH):
    configuration = config.get_config()
    filepath = configuration['datafile_path']
    name_map = configuration['map_test']
    processed_data = DataProcessing.ProcessData(filepath)
    run_model = DataProcessing.RunModel()
    all_actions = []
    all_attentions = []

    cnt_success = 0

    for idx_data, data in enumerate(processed_data.dict_data[tag_split][name_map]):
        # print("data = ", data)
        actions = get_actions(data['action'])
        seq_lang_numpy, seq_world_numpy, seq_action_numpy = processed_data.process_one_data(idx_data, name_map, tag_split)

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

        decoder_hidden = encoder_hidden.view(1, 1, encoder.hidden_size)

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
        all_attentions.append(decoder_attentions)

        #   print "IDX: ", idx_data, "decoded action = ", decoded_actions
        #   print "Instruction : ", data['instruction']
        if actions == decoded_actions:
            cnt_success += 1

    return cnt_success, all_actions, all_attentions[:di+1]


def SampleTest(encoder, decoder, idx_data, sentence, max_length=MAX_LENGTH):
    configuration = config.get_config()
    filepath = configuration['datafile_path']
    name_map = configuration['map_test']
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

    world_state = seq_world[0]
    decoder_input = seq_lang
    decoder_input = decoder_input.cuda() if use_cuda else decoder_input

    decoder_hidden = encoder_hidden.view(1, 1, encoder.hidden_size)

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
    parser = argparse.ArgumentParser(
        description='Testing model ... '
    )
    parser.add_argument(
        '-fp', '--FilePretrain', required=True,
        help='Path of the trained model'
    )
    args = parser.parse_args()
    assert (args.FilePretrain is not None)
    PATH = args.FilePretrain
    ENCODER_PATH = PATH + "/encoder.pkl"
    DECODER_PATH = PATH + "/decoder.pkl"

    encoder = torch.load(ENCODER_PATH)
    attn_decoder = torch.load(DECODER_PATH)

    cnt_success = 0
    tag_split = 'train'
    cnt, _, _ = evaluate(encoder, attn_decoder, tag_split)
    cnt_success += cnt
    tag_split = 'dev'
    cnt, _, _ = evaluate(encoder, attn_decoder, tag_split)
    cnt_success += cnt

    print "Accuracy : ", (cnt_success/ 1070.0)*100, " %"

    # input_sentence1 = "take left onto the brick patch and go all the way to down until you get to where there are butterflies on the wall"
    input_sentence1 = "take a left onto the red brick and go a ways down until you come to the section with the butterflies on the wall"
    output_actions, attentions = SampleTest(encoder, attn_decoder, 82, input_sentence1)
    showAttention(input_sentence1, output_actions, attentions)


if __name__ == '__main__':
    main()
