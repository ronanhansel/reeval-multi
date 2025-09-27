import pandas as pd
import os

DATA_DIR = "../data"

results_df = pd.read_csv(os.path.join(DATA_DIR, "mirt_comparison_repeated.csv"))
results_df.sort_values(by=["K", "Test AUC"], ascending=False).groupby("K").head(1)

import pandas as pd

resmat = pd.read_pickle("../data/resmat.pkl")

import numpy as np
import os
from load_params import load_and_rotate

k = 2; r = 87

theta_se = np.load(f"../output/standard_errors_k{k}.npz")['theta_se']
theta, a, b = load_and_rotate(os.path.join(DATA_DIR, f"mirt_model_k{k}_rep{r}.pt"), rotation=None)

data_for_df = {
    'ability_dim1': theta[:, 0],
    'se_dim1':      theta_se[:, 0],
    'ability_dim2': theta[:, 1],
    'se_dim2':      theta_se[:, 1]
}

theta_df = pd.DataFrame(data_for_df, index=resmat.index)