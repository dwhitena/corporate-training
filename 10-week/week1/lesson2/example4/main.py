import pandas as pd
from sklearn.metrics import accuracy_score

# the data file with observations and predictions
data_file="../data/labeled.csv"

# generate a dataframe based on the data in the file
data = pd.read_csv(data_file)

# calculate the accuracy according to:
# (TP + TN)/(TP + TN + FP + FN)
acc = accuracy_score(data['observation'], data['prediction'])

# print out the accuracy
print("Accuracy: ", acc)

