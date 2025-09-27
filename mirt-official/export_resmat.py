import pandas as pd
import os

os.makedirs("../data/resmat_by_scenario", exist_ok=True)

resmat = pd.read_pickle("../data/resmat.pkl")

for scenario in resmat.columns.get_level_values('scenario').unique():
  print(f"Exporting {scenario}")
  resmat_by_scenario = resmat.loc[:, resmat.columns.get_level_values('scenario') == scenario]
  
  # flatten MultiIndex columns into simple strings (e.g., "domain_item")
  resmat_by_scenario.columns = [f"Item{i}" for i in range(resmat_by_scenario.shape[1])]

  # save to CSV for R
  # optional: reset index if you don’t want model names as rownames in R
  resmat_by_scenario.index = resmat_by_scenario.index.astype(str)
  resmat_by_scenario.to_csv(f"../data/resmat_by_scenario/{scenario}.csv", na_rep="NA")  # R will read "NA" as missing