#!/usr/bin/env python
# -*- coding: utf-8 -*-
#
# Licensed under the GNU LGPL v2.1 - http://www.gnu.org/licenses/lgpl.html


"""

"""

import logging
import sys
import os
from word2vec import Word2Vec, Sent2Vec, LineSentence

logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)
logging.info("running %s" % " ".join(sys.argv))


''' 
LineSentence(input_file): just returns an object.
                          It an iterable that streams the sentences directly from disk/network
size: size/length of the vector (dimensionality of the feature vectors)
window: maximum distance between the current and predicted word within a sentence
sg: defines the training algorithm. By default (`sg=1`), skip-gram is used. Otherwise, `cbow` is employed.
min_count: ignore all words with total frequency lower than this.
workers: use this many worker threads to train the model (=faster training with multicore machines)
'''

input_file = 'GridJellyMultiSent.txt'  # Initially, it was test.txt
model = Word2Vec(LineSentence(input_file), size=100, window=5, sg=1, min_count=5, workers=8)
model.save(input_file + '.model')
model.save_word2vec_format(input_file + '.vec')

sent_file = 'GridJellySingleSent.txt'    # Initially, it was sent.txt
model = Sent2Vec(LineSentence(sent_file), model_file=input_file + '.model')
model.save_sent2vec_format(sent_file + '.vec')

program = os.path.basename(sys.argv[0])
logging.info("finished running %s" % program)