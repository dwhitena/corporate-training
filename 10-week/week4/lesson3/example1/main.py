import pandas as pd
import matplotlib.pyplot as plt

# import the data
data = pd.read_csv('../data/fleet_data.csv')

# get the summary statistics
print(data.describe())

# create histograms of the features
data[['Distance_Feature', 'Speeding_Feature']].hist()
plt.show()

