# Multi level aligner based model

This model uses multi level aligner to generate the context vector. The base paper of this project can be found [here](http://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/download/12522/12021).

Language used : Python 2.7

## Prerequisites

* [PyTorch](https://pytorch.org/) - Computational graphs (which are dynamic in nature) are built on PyTorch
* [PyGame](https://www.pygame.org/wiki/GettingStarted) - Simulation of virtual environment is done using PyGame
* [ArgParse](https://docs.python.org/2/howto/argparse.html) - Command line parsing in Python

## Instructions

For training the model, **two strategies** are followed:

### Strategy 1: Training the model on two maps and testing explicitly on the third map.

* For training: "grid" and "jelly" maps(virtual environments) are used

* For testing: "l" map is used 

### Train model
```
python LSTMmain.py 
```

### Test model

```
python test.py -fp PATH
```
**PATH** stands for the path to the folder containing **encoder.pkl** and **decoder.pkl** which lies within the _tracks_ folder. This folder is created by the end of training.

### Strategy 2: Training and testing the model by using 3 fold cross validation.
```
python LSTMthreefold.py
```

The SAIL route instruction dataset has been downloaded from [here](http://www.cs.utexas.edu/users/ml/clamp/navigation/).
