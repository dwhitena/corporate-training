import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# import the data
data = pd.read_csv('../data/Advertising.csv')
    
# scale the feature and response
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(data[['TV', 'Sales']])
data_scaled_df = pd.DataFrame(data_scaled, columns=['TV', 'Sales'])

# calculate modeled values for Sales based on our 
# linear regression model
slope = 0.553
intercept = 0.215
data_scaled_df['Model'] = data_scaled_df['TV'].apply(lambda x: slope * x + intercept)

# create the plot
fig, ax = plt.subplots()
ax.plot(data_scaled_df['TV'].values, data_scaled_df['Model'].values, 'k--')
ax.plot(data_scaled_df['TV'].values, data_scaled_df['Sales'].values, 'ro')
plt.xlabel('TV', axes=ax)
plt.ylabel('Sales', axes=ax)

# Only draw spine between the y-ticks
ax.spines['left'].set_bounds(0, 1)

# Hide the right and top spines
ax.spines['right'].set_visible(False)
ax.spines['top'].set_visible(False)

# Only show ticks on the left and bottom spines
ax.yaxis.set_ticks_position('left')
ax.xaxis.set_ticks_position('bottom')

plt.show()

