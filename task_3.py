import k_means
import sklearn.metrics
import matplotlib.pyplot as plt

attributes, list_of_data = k_means.read_csv('mean_data.csv')
k_means.normalize_data(list_of_data)
vals = []

for k in range(2, 100):
	print(k)
	init_centers = k_means.random_centers(k, list_of_data)
	clusters = k_means.k_means(init_centers, list_of_data)
	
	X, y_label = [], []
	for cluster in clusters:
		for data in cluster:
			X.append(data[:-1])
			y_label.append(int(data[-1]))

	y_pred = k_means.get_predictions(clusters)

	# metrics without ground truth
	vals.append(sklearn.metrics.calinski_harabasz_score(X, y_pred))
	

plt.plot(vals)
plt.show()