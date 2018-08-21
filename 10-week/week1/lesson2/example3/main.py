import pandas as pd
from sklearn.metrics import r2_score

# the data file with observations and predictions
data_file="../data/continuous.csv"

# generate a dataframe based on the data in the file
data = pd.read_csv(data_file)

# calculate the R squared
r2 = r2_score(data['observation'], data['prediction'])

# print out the R squared
print("R^2: ", r2)

