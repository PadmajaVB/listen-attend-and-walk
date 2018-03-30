from keras.models import Sequential
from keras.layers import LSTM, Bidirectional
import config
from AttentionDecoder import AttentionDecoder


class SeqToSeq:
	def __init__(self):
		self.model_conf = config.get_model_config()
		self.dim_lang = self.model_conf['dim_lang']
		self.dim_world = self.model_conf['dim_world']
		self.dim_action = self.model_conf['dim_action']
		self.optimizer = self.model_conf['optimizer']
		self.dropout_rate = self.model_conf['dropout_rate']
		self.beam_size = self.model_conf['beam_size']
		self.dim_model = self.model_conf['dim_lstm_model']

		self.conf = config.get_config()
		self.epoch = self.conf['max_epochs']

	def build_model(self, X1, X2, y):
		model = Sequential()
		n_features = 100
		model.add(LSTM(self.dim_model, input_shape=(524, n_features), return_sequences=True))
		model.add(AttentionDecoder(100, n_features))
		model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

		for epoch in range(self.epoch):
			model.fit(X1, y)
