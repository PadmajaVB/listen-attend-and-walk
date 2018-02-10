import pickle
import pandas as pd
import csv
import json
import numpy
import pprint as pp


with open('mapscap1000.pickle', 'r') as f:
    devset = pickle.load(f)
    
'''
for key in devset[0]:
	print key
prints "nodes, edges and name"
'''

for key in devset[2]:
	print key
	print type(key)
	print 
	print devset[2][key]
	print type(devset[2][key])
	print
	print




# print devset[0]['nodes']

# x = {'planet' : {'has': {'plants': 'yes', 'animals': 'yes', 'cryptonite': 'no'}, 'name': 'Earth'}}

# y = {'key': [['p',2],[3,1]], 'k': 'p'}

# pp.pprint(y, indent=4)

# print type(x)

# print json.dumps(devset[0], indent=2)

# print json.dumps(y, indent=2)

# for key in devset[0]:
    # print "key = ",key
    # print "value = ", devset[0][key], "\n\n"

# df = pd.DataFrame.from_dict(devset[0], orient="index")
# df.to_csv("mapscap1000-1.csv")
    # count += 1
    
# print "devset : ", type(devset)
    
# df = pd.DataFrame.from_dict(devset, orient="index")

# df.to_csv("mapscap1000.csv")

# with open("mapscap1000.csv", "w") as f:
   # w = csv.writer(f)
   # w.writerows(devset)
