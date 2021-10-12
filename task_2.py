import k_means
import sklearn.metrics
import matplotlib.pyplot as plt

attributes, list_of_data = k_means.read_csv('mean_data.csv')
k_means.normalize_data(list_of_data)
vals = []
for k in range(1, 200):
	print(k)
	init_centers = k_means.random_centers(k, list_of_data)
	clusters = k_means.k_means(init_centers, list_of_data)
	y_label, y_pred = k_means.get_predictions(clusters)
	vals.append(sklearn.metrics.homogeneity_score(y_label, y_pred))

plt.plot(vals)
plt.show()