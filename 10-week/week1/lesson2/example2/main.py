import math
import pandas as pd

# the data file with observations and predictions
data_file="../data/continuous.csv"

# generate a dataframe based on the data in the file
data = pd.read_csv(data_file)

# calculate the RMSE
data['squared_error'] = (data['observation'] - data['prediction'])**2
mse = data['squared_error'].mean()
rmse = math.sqrt(mse)

# print out the RMSE
print("RMSE: ", rmse)

