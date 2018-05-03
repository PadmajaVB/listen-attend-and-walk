import pickle
import pandas as pd


with open('stat.pickle', 'r') as f:
    x = pickle.load(f)
    
df = pd.DataFrame.from_dict(x, orient="index")

df.to_csv("stat.csv")

