from numpy.core.numeric import full
from scipy.sparse.construct import rand
import k_means, random
from sklearn import metrics

attributes, list_of_data = k_means.read_csv('mean_data.csv')
k_means.normalize_data(list_of_data)

# get the value from part iii)
k = 50

def evaluate_k_means_with_k_centers(full_data, centers_ind)->dict:
	indices = []
	init_centers = []

	for i in range(len(full_data)):
		if i in centers_ind:
			init_centers.append(full_data[i][:-1])
		else:
			indices.append(i)

	# metrics
	NMI_sc, ARI_sc, Hom_sc = 0, 0, 0

	size = len(list_of_data)
	test_size = int(0.2*(size - k))
	
	for _ in range(50):
		train_data, test_data = [], []
		indices_test_points = random.sample(indices, test_size)

		for i in range(size):
			if i in indices_test_points:
				test_data.append(list_of_data[i])
			else:
				train_data.append(list_of_data[i])

		clusters = k_means.k_means(init_centers, train_data)
		y_pred = k_means.get_predictions_for_test(test_data, clusters)
		y_label = [test_data[i][-1] for i in range(test_size)]

		NMI_sc += metrics.normalized_mutual_info_score(y_label, y_pred)
		ARI_sc += metrics.adjusted_rand_score(y_label, y_pred)
		Hom_sc += metrics.homogeneity_score(y_label, y_pred)

	NMI_sc /= 50
	ARI_sc /= 50
	Hom_sc /= 50

	return [NMI_sc, ARI_sc,Hom_sc]

def random_k_centers(k, full_data):
	metrics = [[],[],[]]
	for _ in range(50):
		indices = [i for i in range(len(full_data))]
		chosen_centers = random.sample(indices, k)
		ps_metrics = evaluate_k_means_with_k_centers(full_data, chosen_centers)
		for i in range(3):
			metrics[i].append(ps_metrics[i])
	
	import matplotlib.pyplot as plt
	figure, ax = plt.subplots()
	
	for i in range(3):
		ax.plot(metrics[i])

	plt.show()

random_k_centers(k, list_of_data)

		






