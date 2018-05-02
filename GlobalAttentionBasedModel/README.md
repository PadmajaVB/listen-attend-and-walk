# Global Attention Based Model

This model uses global attention mechanism to generate the context vector. Further architectural details of the model can be found [here](https://arxiv.org/pdf/1508.04025.pdf)

Language used : Python 2.7

## Prerequisites

* [PyTorch](https://pytorch.org/) - Computational graphs (which are dynamic in nature) are built on PyTorch
* [PyGame](https://www.pygame.org/wiki/GettingStarted) - Simulation of virtual environment is done using PyGame
* [ArgParse](https://docs.python.org/2/howto/argparse.html) - Command line parsing in Python

## Instructions

### Train model

```
python main.py 
```

### Test model

```
python test.py -fp PATH
```
**PATH** stands for the path to the folder containing **encoder.pkl** and **decoder.pkl** which lies within the _tracks_ folder. This folder is created by the end of training.

The SAIL route instruction dataset has been downloaded from [here](http://www.cs.utexas.edu/users/ml/clamp/navigation/).

