import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_pacf

# import the data
data = pd.read_csv('../data/AirPassengers.csv', index_col=0)

# plot the PACF
plot_pacf(data.set_index('time')['value'], lags=20)
plt.show()
