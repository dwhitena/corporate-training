import pandas as pd
from pandas.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np

# 1. Import and parse the dataset.
data = pd.read_csv('diabetes.csv')
print(data.head())

# 2. Print out summary stats.
print(data.describe())

# 3. Create (i) histogram plots for the data in 
# each column, and (ii) scatter plots showing 
# the correlation between columns.
scatter_matrix(data, alpha=0.2, figsize=(10, 10))
plt.show()

# 4. Split the data into training, test, and 
# holdout data sets
mask = np.random.rand(len(data)) < 0.8
training = data[mask]
test = data[~mask]

mask = np.random.rand(len(training)) < 0.8
holdout = training[~mask]
training = training[mask]

# save these data sets
training.to_csv('training.csv', index=False)
test.to_csv('test.csv', index=False)
holdout.to_csv('holdout.csv', index=False)
