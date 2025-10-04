import pandas as pd
import numpy as np

df = pd.read_csv("./data/hotpot_train.csv")
train_1, train_2, train_3, train_4, train_5, train_6 = np.array_split(df, 6)

train_1.to_csv("./data/hotpot_train_1.csv", index=False)
train_2.to_csv("./data/hotpot_train_2.csv", index=False)
train_3.to_csv("./data/hotpot_train_3.csv", index=False)
train_4.to_csv("./data/hotpot_train_4.csv", index=False)
train_5.to_csv("./data/hotpot_train_5.csv", index=False)
train_6.to_csv("./data/hotpot_train_6.csv", index=False)
