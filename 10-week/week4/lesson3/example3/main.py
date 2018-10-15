import pandas as pd
from sklearn.cluster import KMeans
from sklearn import metrics
import matplotlib.pyplot as plt

# import the data
data = pd.read_csv('../data/fleet_data.csv')

# define the k-means model
model = KMeans(n_clusters=2)

# "fit" the model, which in this case just
# means finding the cluster centroids
predictions = model.fit_predict(data[['Distance_Feature', 'Speeding_Feature']])

# evaluate the clusters
score = metrics.silhouette_score(data[['Distance_Feature', 'Speeding_Feature']], 
        model.labels_, metric='euclidean')
print('Silhouette Score: ', score)

# scatter plot
plt.scatter(data['Distance_Feature'].values, data['Speeding_Feature'].values, c=predictions)
plt.title("Detected Clusters")
plt.xlabel('Distance')
plt.ylabel('Speeding')
plt.show()
