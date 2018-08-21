import pandas as pd
from sklearn.metrics import recall_score

# the data file with observations and predictions
data_file="../data/labeled.csv"

# generate a dataframe based on the data in the file
data = pd.read_csv(data_file)

# calculate the recall according to:
# TP/(TP + FN)
recall = recall_score(data['observation'], data['prediction'], average=None)

# print out the recall
print("Recall: ", recall)

