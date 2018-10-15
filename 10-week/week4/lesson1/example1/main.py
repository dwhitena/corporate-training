import pandas as pd

data = pd.read_csv('../data/AirPassengers.csv', index_col=0)
print(data.head())
