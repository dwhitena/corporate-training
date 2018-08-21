import pandas as pd
from sklearn.metrics import precision_score

# the data file with observations and predictions
data_file="../data/labeled.csv"

# generate a dataframe based on the data in the file
data = pd.read_csv(data_file)

# calculate the precision according to:
# TP/(TP + FP)
ppv = precision_score(data['observation'], data['prediction'], average=None)

# print out the precision
print("Precision: ", ppv)

