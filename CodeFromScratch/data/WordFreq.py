import pickle
import pandas as pd


with open('stat.pickle', 'r') as f:
    x = pickle.load(f)
count = 0
for val in x['wordfreq']:
	print val
	count += val

print count
