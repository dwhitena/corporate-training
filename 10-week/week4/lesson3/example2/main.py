import pandas as pd
from sklearn.cluster import KMeans

# import the data
data = pd.read_csv('../data/fleet_data.csv')

# define the k-means model
model = KMeans(n_clusters=2)

# "fit" the model, which in this case just
# means finding the cluster centroids
predictions = model.fit_predict(data[['Distance_Feature', 'Speeding_Feature']])

# print the centroids
print(model.cluster_centers_)
