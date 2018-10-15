import pandas as pd
import matplotlib.pyplot as plt

# import the data
data = pd.read_csv('../data/AirPassengers.csv', index_col=0)

# plot the series
data.set_index('time').plot()
plt.show()
