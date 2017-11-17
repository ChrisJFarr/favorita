# todo create simple prediction model and make submission

import pandas as pd
import numpy as np
import pickle

train = pd.read_csv("data/train.csv")

# Dumping into pickle, might be a bit faster on upload
file_name = "data/train.pkl"
pickle.dump(train, open(file_name, "wb"))




