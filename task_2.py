import k_means
from sklearn import metrics

attributes, list_of_data = k_means.read_csv('mean_data.csv')
k_means.normalize_data(list_of_data)
print("Enter the value of k:", end=' ')
k = int(input())

init_centers = k_means.random_centers(k, list_of_data)
clusters = k_means.k_means(init_centers, list_of_data)
y_pred = k_means.get_predictions_for_clusters(clusters)

X, y_label = [], []
for cluster in clusters:
	for data in cluster:
		X.append(data[:-1])
		y_label.append(int(data[-1]))

# metrics with ground truth
print(metrics.homogeneity_score(y_label, y_pred))
print(metrics.normalized_mutual_info_score(y_label, y_pred))
print(metrics.adjusted_rand_score(y_label, y_pred))

# metrics without ground truth
print(metrics.silhouette_score(X, y_pred))
print(metrics.calinski_harabasz_score(X, y_pred))