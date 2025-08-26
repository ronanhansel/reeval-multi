import numpy as np
import pandas as pd

# Run this once to create the CSV files
item_factors = np.load("../data/all_item_factors.npy")
subject_scores = np.load("../data/subject_scores.npy")

pd.DataFrame(item_factors).to_csv("../data/all_item_factors.csv", index=False)
pd.DataFrame(subject_scores).to_csv("../data/subject_scores.csv", index=False)