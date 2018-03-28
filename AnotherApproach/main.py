import config
import DataProcessing
from gensim.models import Word2Vec


def main():
	# TODO preprocess the input file to get standard vectors
	configuration=config.get_config()
	filepath = configuration['datafile_path']
	word_embedding = configuration['word_embedding']
	processed_data = DataProcessing.ProcessData(filepath)
	word_vectors = Word2Vec.load(word_embedding)
	print "Word embeddings: ", word_vectors

	""" Model designing part """
	# TODO design encoder


# TODO design multi-level aligner

# TODO design decoder

""" Training the model """
# TODO write training code

# TODO map the o/p action sequence to the i/p instruction for effective backpropagation

""" Testing the model """
# TODO write testing code

# TODO simulate the output
