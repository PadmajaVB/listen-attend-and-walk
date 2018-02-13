import numpy as np
from keras.models import Model
from keras.layers import Bidirectional, Input
from keras.layers import LSTM
import multilevel_aligner as ma


vectors = np.loadtxt(open("./Preprocessing/GridJellySingleSent.csv", "rb"), delimiter=",", skiprows=1)

file_len = 10
input_vectors = []

for file_idx in range(file_len):
    input_vectors.append(vectors[file_idx].reshape(1, 100))

input_vectors = np.array(input_vectors)

inputs1 = Input(shape=(1, 100))
lstm_high_level_vectors = Bidirectional(LSTM(100, return_sequences=True))(inputs1)
# 100 corresponds to the number of outputs we want
model = Model(inputs=inputs1, outputs=[lstm_high_level_vectors])

high_level_rep = model.predict(input_vectors)

context_vectors = []

for i in range(file_len):
    c_vector = ma.generate_context_vector(input_vectors[0], high_level_rep[0])
    context_vectors.append(c_vector)

context_vectors = np.array(context_vectors)
print "Length = ", len(context_vectors)

print "Shape of input vectors = ", input_vectors.shape
print "Shape of hidden vectors = ", high_level_rep.shape
print "Shape of context vectors = ", context_vectors.shape

'''
Sample output:
Shape of input vectors =  (10, 1, 100)
Shape of hidden vectors =  (10, 1, 200)
Shape of context vectors =  (10, 1, 300)
'''
