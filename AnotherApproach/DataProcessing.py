import pickle
import numpy
import config

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
