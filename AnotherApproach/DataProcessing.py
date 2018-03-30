import pickle
import numpy
import config
from gensim.models import Word2Vec

dtype = numpy.float32


class ProcessData(object):
    def __init__(self, datafile_path):
        #
        print "Initializing data pre-processing ....."

        assert (datafile_path is not None)
        self.datafile_path = datafile_path

        #
        #
        with open(self.datafile_path + 'databag3.pickle', 'r') as f:
            raw_data = pickle.load(f)
        with open(self.datafile_path + 'valselect.pickle', 'r') as f:
            val_set = pickle.load(f)
        with open(self.datafile_path + 'stat.pickle', 'r') as f:
            stats = pickle.load(f)
        with open(self.datafile_path + 'mapscap1000.pickle', 'r') as f:
            self.maps = pickle.load(f)
            # maps is a list
        #
        self.lang2idx = stats['word2ind']
        self.dim_lang = stats['volsize']  # 524
        #
        self.configuration = config.get_model_config()
        self.dim_world = self.configuration['dim_world']
        self.dim_action = self.configuration['dim_action']

        self.names_map = ['grid', 'jelly', 'l']

        #
        self.dict_data = {
            'train': {},
            'dev': {}
        }
        #

        """Grid-874 instructions, Jelly-1293 instructions, L-1070 instructions"""
        for name_map in self.names_map:
            self.dict_data['train'][name_map] = []
            self.dict_data['dev'][name_map] = []
            for idx_data, data in enumerate(raw_data[name_map]):
                if idx_data in val_set[name_map]:
                    """100 instructions per map"""
                    self.dict_data['dev'][name_map].append(data)
                else:
                    self.dict_data['train'][name_map].append(data)
        #
        self.map2idx = {
            'grid': 0, 'jelly': 1, 'l': 2
        }
        self.idx2map = {
            0: 'grid', 1: 'jelly', 2: 'l'
        }
        #
        self.seq_lang_numpy = None
        self.seq_world_numpy = None
        self.seq_action_numpy = None
        #

    def get_left_and_right(self, direc_current):
        # direc_current can be 0 , 90, 180, 270
        # it is the current facing direction
        assert(direc_current == 0 or direc_current == 90 or direc_current == 180 or direc_current == 270)
        left = direc_current - 90
        if left == -90:
            left = 270
        right = direc_current + 90
        if right == 360:
            right = 0
        behind = direc_current + 180
        if behind == 360:
            behind = 0
        elif behind == 450:
            behind = 90
        return left, right, behind


    #
    def get_pos(self, idx_data, name_map, tag_split):
        one_data = self.dict_data[tag_split][name_map][idx_data]
        path_one_data = one_data['cleanpath']
        return path_one_data[0], path_one_data[-1]  # starting and ending positions of the form (x,y,orientation)

    def process_one_data(self, idx_data, name_map, tag_split):
        # process the data with id = idx_data
        # in map[name_map]
        # with tag = tag_split, i.e., 'train' or 'dev'
        one_data = self.dict_data[tag_split][name_map][idx_data]

        """ list of word indices """
        self.list_word_idx = [self.lang2idx[w] for w in one_data['instruction'] if w in self.lang2idx]


        self.conf = config.get_config()
        word_embedding = self.conf['word_embedding']
        word_vectors = Word2Vec.load(word_embedding)

        seq_lang = []

        for word in one_data['instruction']:
            seq_lang.append(word_vectors[word])

        self.seq_lang_numpy = numpy.array(seq_lang)

        """ Zero matrix of dim (len(one_data['cleanpath'])*78)"""
        self.seq_world_numpy = numpy.zeros(
            (len(one_data['cleanpath']), self.dim_world),
            dtype=dtype
        )

        """ idx_map is an index of the map i.e grid-0, jelly-1, l-2 """
        idx_map = self.map2idx[
            one_data['map'].lower()
        ]
        nodes = self.maps[idx_map]['nodes']
        for idx_pos, pos in enumerate(one_data['cleanpath']):
            x_current, y_current, direc_current = pos[0], pos[1], pos[2]
            #
            count_pos_found = 0
            #
            for idx_node, node in enumerate(nodes):
                if node['x'] == x_current and node['y'] == y_current:
                    # find this position in the map
                    # so we can get its feature
                    count_pos_found += 1
                    #
                    left_current, right_current, behind_current = self.get_left_and_right(direc_current)
                    '''
                    note:
                    for node, we keep it as [0,..,1,..,0] one hot
                    but for all directions, the last entry of feature tags if this way is walkable: 0
                    1 -- it is blocked
                    '''
                    feat_node = numpy.cast[dtype](
                        node['objvec']
                    )
                    feat_forward = numpy.cast[dtype](
                        node['capfeat'][direc_current]
                    )
                    feat_left = numpy.cast[dtype](
                        node['capfeat'][left_current]
                    )
                    feat_right = numpy.cast[dtype](
                        node['capfeat'][right_current]
                    )
                    feat_behind = numpy.cast[dtype](
                        node['capfeat'][behind_current]
                    )
                    self.seq_world_numpy[idx_pos, :] = numpy.copy(
                        numpy.concatenate(
                            (feat_node, feat_forward, feat_left, feat_right, feat_behind),
                            axis=0
                        )
                    )
            assert(count_pos_found > 0)
            # have to find this position in this map
        #
        self.seq_action_numpy = numpy.zeros(
            (len(one_data['action']), ),
            dtype=numpy.int32
        )

        """numpy array of index at which the one-hot action vector value is 1"""
        for idx_action, one_hot_vec_action in enumerate(one_data['action']):
            self.seq_action_numpy[idx_action] = numpy.argmax(
                one_hot_vec_action
            )
        # print "self.seq_action_numpy.shape=",self.seq_action_numpy.shape
        # finished processing !
        return self.seq_lang_numpy, self.seq_world_numpy, self.seq_action_numpy