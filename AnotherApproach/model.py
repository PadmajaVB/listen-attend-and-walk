from keras.models import Sequential
from keras.layers import LSTM
import config

class SeqToSeq:
	def __init__(self):
		model_conf = config.get_model_config()
		self.dim_lang = model_conf['dim_lang']
		self.dim_world = model_conf['dim_world']
		self.dim_action = model_conf['dim_action']
		self.optimizer = model_conf['optimizer']
		self.dropout_rate = model_conf['dropout_rate']
		self.beam_size = model_conf['beam_size']
		self.dim_model = model_conf['dim_lstm_model']

	def build_model(self):

		model = Sequential()
		input_shape = (None, 100)
		model.add(LSTM(self.dim_model, input_shape, return_sequences=True))

