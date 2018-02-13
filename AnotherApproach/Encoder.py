import numpy as np

from keras.models import Model
from keras.layers import Bidirectional, Input
from keras.layers import LSTM


filePath = "./Preprocessing/"

vectors = np.loadtxt(open("GridJellySingleSent.csv", "rb"), delimiter=",", skiprows=1)

print "vectors.shape = ", vectors.shape

print "vectors[0] = ", vectors[0]

file_len = 10
input_vectors = []

print "file_len = ", file_len

for file_idx in range(file_len):
    input_vectors.append(vectors[file_idx].reshape(1, 100))

input_vectors = np.array(input_vectors)

print "input_vectors[0] = ", input_vectors[0]

inputs1 = Input(shape=(1, 100))
lstm_outputs = Bidirectional(LSTM(100, return_sequences=True))(inputs1)
# 100 corresponds to the number of outputs we want
model = Model(inputs=inputs1, outputs=[lstm_outputs])

outputs = model.predict(input_vectors)

print
print len(outputs)
print outputs.shape
print outputs[0].shape


