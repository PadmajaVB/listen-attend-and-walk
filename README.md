# listen-attend-and-walk

Converting natural language instructions into its corresponding action sequence.  

Here is the [link](https://arxiv.org/abs/1506.04089) to the base paper that is being referred. 

Language used : Python 2.7

## Prerequiisites

* [Anaconda](https://www.continuum.io/) - Anaconda includes all the Python-related dependencies
* [Theano](http://deeplearning.net/software/theano/) - Computational graphs are built on Theano
* [ArgParse](https://docs.python.org/2/howto/argparse.html) - Command line parsing in Python

## Instructions

### Train model

```
python train_model.py 
```

### Test model

```
python test_model.py -fp PATH
```
**PATH** stands for the path to the file **model.pkl** which lies within the _tracks_ folder. This folder is created by the end of training.  

The SAIL route instruction dataset has been downloaded from [here](http://www.cs.utexas.edu/users/ml/clamp/navigation/).

The following [code](https://github.com/klb3713/sentence2vec) has been used for embedding sentences to vectors. It uses the skip-gram model.

This project is built using Hongyuan Mei's [code](https://github.com/HMEIatJHU/NeuralWalker) as base. 
