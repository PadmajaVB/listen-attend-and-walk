Training the model:


1) Training on two maps and testing explicitly on third map

$ python LSTMmain.py

$ python LSTMtest.py -fp "\<path to trained model\>"


2) Training with 3 fold cross validation

$ python LSTMthreefold.py
