from gensim.models import Word2Vec
import re


def loadSentences(filepath):
	# count = 0
	all_sentences = []
	with open(filepath) as fp:
		for line in fp:
			line = line.lower()
			splitted_line = re.split(',| |, |\n', line)
			splitted_line = [x for x in splitted_line if x != '']
			all_sentences.append(splitted_line)
	for x in all_sentences:
		print x

	print len(all_sentences)
	print type(all_sentences)
	return all_sentences


def trainModel(filepath):
	sentences = loadSentences(filepath)
	print "type(sentences) = ", sentences[0]
	# train model
	model = Word2Vec(sentences, size=100, window=5, sg=1, min_count=1, workers=3)
	# summarize the loaded model
	print model
	# summarize vocabulary
	words = list(model.wv.vocab)
	print len(words)
	# access vector for one word
	print model['carpet']
	model.wv.save_word2vec_format('WordEmbeddings.txt', binary=False)
	# save model
	model.save('WordEmbeddings.bin')
	# load model
	new_model = Word2Vec.load('WordEmbeddings.bin')
	print new_model


if __name__ == '__main__':
	filepath = "allMapsSingleSent.txt"
	trainModel(filepath)