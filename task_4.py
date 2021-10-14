import k_means, random
from sklearn import metrics

attributes, list_of_data = k_means.read_csv('mean_data.csv')
k_means.normalize_data(list_of_data)

# get the value from part iii)
k = 50

def get_variance(data: list):
	mean_data = 0
	mean_data2 = 0
	for val in data:
		mean_data += val
		mean_data2 += val**2
	mean_data2 /= len(data)
	mean_data /= len(data)
	var_data = mean_data2 - mean_data**2
	return var_data

def evaluate_k_means_with_k_centers(full_data, centers_ind: set)->dict:
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
		clust_test_ind, y_pred = k_means.get_clusters_and_preds_for_test(test_data, clusters)
		y_label = [test_data[i][-1] for i in range(test_size)]

		NMI_sc += metrics.normalized_mutual_info_score(y_label, y_pred)
		ARI_sc += metrics.adjusted_rand_score(y_label, y_pred)
		Hom_sc += metrics.homogeneity_score(y_label, y_pred)

	NMI_sc /= 50
	ARI_sc /= 50
	Hom_sc /= 50

	return [NMI_sc, ARI_sc,Hom_sc]

def evaluate_init(k: int, full_data: list, method, method_name)->None:
	metrics = [[],[],[]]
	for _ in range(50):
		ps_metrics = evaluate_k_means_with_k_centers(full_data, method(k, full_data))
		for i in range(3):
			metrics[i].append(ps_metrics[i])
	
	import matplotlib.pyplot as plt
	figure, ax = plt.subplots()
	colors = ['blue', 'red', 'green']

	for i in range(3):
		ax.plot(metrics[i], color=colors[i])
	print("\nWith", method_name + ", the variance of metrics were:")
	print("The variance of NMI:", get_variance(metrics[0]))
	print("The variance of ARI:", get_variance(metrics[1]))
	print("The variance of Homogeneity:", get_variance(metrics[2]))
	plt.savefig(method_name+'.png')
	plt.close()

def heuristic_based_init(k, full_data):
	indices = [i for i in range(len(full_data))]
	c0 = random.choice(indices)
	centers = {c0}

	for centr_id in range(k - 1):
		max_dist = 0
		ps_center = -1
		for data_id in range(len(full_data)):
			if data_id in centers:
				continue
			min_dis = k_means.get_dist(full_data[c0][:-1], full_data[data_id][:-1])
			for prev_id in centers:
				min_dis = min(min_dis, k_means.get_dist(full_data[prev_id][:-1], full_data[data_id][:-1]))
			if min_dis >= max_dist:
				max_dist = min_dis
				ps_center = data_id
		centers.add(ps_center)
	return centers

def random_center_init(k, full_data):		
	indices = [i for i in range(len(full_data))]
	chosen_centers = set(random.sample(indices, k))
	return chosen_centers

evaluate_init(k, list_of_data, random_center_init, "random_center_init")
evaluate_init(k, list_of_data, heuristic_based_init, "heuritics_based_init")








