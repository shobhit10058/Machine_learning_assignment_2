import k_means
from sklearn import metrics
import matplotlib.pyplot as plt

attributes, list_of_data = k_means.read_csv('mean_data.csv')
k_means.normalize_data(list_of_data)
tot_variance = []
Sill_Sc = []
K_range = []

for k in range(2, 200):
	print("K =",k,"done")
	init_centers = k_means.random_centers(k, list_of_data)
	clusters = k_means.k_means(init_centers, list_of_data)
	
	X, y_label = [], []
	for cluster in clusters:
		for data in cluster:
			X.append(data[:-1])
			y_label.append(int(data[-1]))

	y_pred = k_means.get_predictions_for_clusters(clusters)

	# metrics without ground truth
	tot_variance.append(k_means.get_total_variance(clusters))
	Sill_Sc.append(metrics.silhouette_score(X, y_pred))
	K_range.append(k)

plt.plot(K_range, tot_variance)
plt.savefig("tot_variance_vs_k.png")
print("generated tot_variance_vs_k.png")
plt.close()
plt.plot(K_range, Sill_Sc)
plt.savefig("Sillhouette_Score_vs_k.png")
print("generated Sillhouette_Score_vs_k.png")
plt.close()