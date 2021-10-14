import random
import k_means
from sklearn import metrics
import matplotlib.pyplot as plt

attributes, list_of_data = k_means.read_csv('mean_data.csv')
k_means.normalize_data(list_of_data)
tot_variance = []
Sill_Sc = []
wang_sc = []
K_range = []

def wang_cross_validation(k, c, full_data: list):
	dearr = 0
	for _ in range(c):
		ps_data = full_data.copy()
		random.shuffle(ps_data)
		m = int((3/8)*len(full_data))
		s1 = ps_data[:m]
		s2 = ps_data[m:2*m]
		s3 = ps_data[2*m:]
		cluster_s1 = k_means.k_means(k_means.random_centers(k, s1), s1)
		cluster_s2 = k_means.k_means(k_means.random_centers(k, s2), s2)
		clust_s3_s1, pred_s1 = k_means.get_clusters_and_preds_for_test(s3, cluster_s1)
		clust_s3_s2, pred_s2 = k_means.get_clusters_and_preds_for_test(s3, cluster_s2)
		for it_1 in range(len(s3)):
			for it_2 in range(len(s3)):
				dearr += ((clust_s3_s1[it_1] == clust_s3_s2[it_1]) != (clust_s3_s1[it_2] == clust_s3_s2[it_2]))
	dearr /= (2*c)
	return dearr

def plot_metric(x, y, name):
	plt.plot(x, y)
	plt.savefig(name+".png")
	print("generated",name+".png")
	plt.close()

for k in range(2, 100):
	print("K =",k,"done")
	init_centers = k_means.random_centers(k, list_of_data)
	clusters = k_means.k_means(init_centers, list_of_data)
	
	X, y_label = [], []
	for cluster in clusters:
		for data in cluster:
			X.append(data[:-1])
			y_label.append(int(data[-1]))

	y_pred = k_means.get_predictions_for_clusters(clusters)

	tot_variance.append(k_means.get_total_variance(clusters))
	Sill_Sc.append(metrics.silhouette_score(X, y_pred))
	wang_sc.append(wang_cross_validation(k, 20, list_of_data))
	K_range.append(k)

plot_metric(K_range, wang_sc, "wang_method")
plot_metric(K_range, Sill_Sc, "Sill_Sc")
plot_metric(K_range, tot_variance, "tot_variance")