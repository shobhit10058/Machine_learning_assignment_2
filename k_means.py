import random

def read_csv(file: str)->tuple:
	list_of_data = []
	with open(file) as f:
		attributes = list(f.readline().split(','))
		while True:
			curr_row = f.readline()
			if len(curr_row) == 0:
				break
			ps_vals = list(map(float, curr_row.split(',')))
			list_of_data.append(ps_vals)
	return (attributes, list_of_data)

def normalize_data(list_of_data: list):
	data_size = len(list_of_data)
	for attr_ind in range(len(list_of_data[0])):
		mx_val = 0
		for row in range(data_size):
			mx_val = max(mx_val, list_of_data[row][attr_ind])
		for row in range(data_size):
			list_of_data[row][attr_ind] /= mx_val

def get_dist(point_1: list, point_2: list) -> float:
	curr_dis = 0
	for attr_ind in range(len(point_1)):
			curr_dis += ((point_1[attr_ind] - point_2[attr_ind])**2)
	return curr_dis**0.5

# returns the index of closest center
def closest_centre(centers: list, data: list)->int:
	closest_ind = -1
	min_dis = -1
	for ind in range(len(centers)):
		curr_dis = get_dist(centers[ind], data)
		if min_dis == -1 or curr_dis <= min_dis:
			closest_ind = ind
			min_dis = curr_dis
	return closest_ind

def get_curr_cluster(centers: list, list_of_data: list)->list:
	k = len(centers)
	clusters = [[] for _ in range(k)]
	for curr_data in list_of_data:
		clusters[closest_centre(centers, curr_data)].append(curr_data)
	return clusters

def get_center(cluster: list)->list:
	size_of_cluster = len(cluster)
	assert(size_of_cluster > 0) 	
	dim = (len(cluster[0]) - 1)
	center = [0 for _ in range(dim)]
	for curr_data in cluster:
		# assert(len(curr_data) == 9)
		for attr_ind in range(dim):
			center[attr_ind] += curr_data[attr_ind]
	for attr_ind in range(dim):
		center[attr_ind] /= size_of_cluster
	return center

def get_centers(clusters: list)->list:
	centers = []
	for cluster in clusters:
		centers.append(get_center(cluster))
	return centers

def random_centers(k, list_of_data):
	init_centers = random.sample(list_of_data, k)
	for ind in range(k):
		init_centers[ind] = list(init_centers[ind])
		init_centers[ind].pop()
	return init_centers

# give the optional value of tolerance if 
# convergence is taking too much time, by
# default it is zero
def k_means(init_centers: list, list_of_data: list, tol = 0)->list:
	k = len(init_centers)
	centers = [centre for centre in init_centers]
	it = 0
	while True:
		it += 1
		clusters = get_curr_cluster(centers, list_of_data)
		new_centers = get_centers(clusters)		
		mx_dist = 0
		for ind in range(len(centers)):
			mx_dist = max(mx_dist, get_dist(new_centers[ind], centers[ind]))
		if mx_dist <= tol:
			break
		centers = [centre for centre in new_centers]
	# print("The centers are:")
	# for centre in centers:
	# 	print(centre)
	# print("Total number of iteration: ", it)
	clusters = get_curr_cluster(centers, list_of_data)
	return clusters

def get_pred_for_cluster(cluster: list):
	pred = 0
	lab_counts = {0: 0, 1: 0}
	for curr_data in cluster:
		lab_counts[int(curr_data[-1])] += 1
	if lab_counts[0] > lab_counts[1]:
		return 0
	return 1

def get_predictions_for_clusters(clusters: list):
	y_pred = []
	for cluster in clusters:
		pred = get_pred_for_cluster(cluster)
		ps_siz = len(cluster)
		y_pred.extend([pred for _ in range(ps_siz)])
	return y_pred

def get_predictions_for_test(test_data: list, clusters: list):
	y_pred = []
	centers = get_centers(clusters)
	predictions_for_clusters = [get_pred_for_cluster(clusters[i]) for i in range(len(clusters))]
	for data in test_data:
		clust_ind = closest_centre(centers, data)
		y_pred.append(predictions_for_clusters[clust_ind])
	return y_pred