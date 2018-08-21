import pandas as pd

# the data file with observations and predictions
data_file="../data/continuous.csv"

# generate a dataframe based on the data in the file
data = pd.read_csv(data_file)

# calculate the MAE
data['absolute_error'] = abs(data['observation'] - data['prediction'])
mae = data['absolute_error'].mean()

# print out the MAE
print("MAE: ", mae)

