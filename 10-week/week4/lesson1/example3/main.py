import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf

# import the data
data = pd.read_csv('../data/AirPassengers.csv', index_col=0)

# plot the ACF
plot_acf(data.set_index('time')['value'])
plt.show()
