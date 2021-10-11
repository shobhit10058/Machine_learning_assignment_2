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

# returns the index of closest center
def closest_centre(centers: list, data: list)->int:
	closest_ind = -1
	min_dis_sq = 0
	for ind in range(len(centers)):
		center = centers[ind]
		curr_dis_sq = 0
		for attr_ind in range(len(center)):
				curr_dis_sq += ((data[attr_ind] - center[attr_ind])**2)
		if curr_dis_sq <= min_dis_sq:
			closest_ind = ind
	return closest_ind

def get_curr_cluster(centers: list, list_of_data: list)->list:
	k = len(centers)
	clusters = [[] for _ in range(k)]
	for curr_data in list_of_data:
		clusters[closest_centre(centers, curr_data)].append(curr_data)
	return clusters

def get_center(cluster: list)->dict:
	center = []
	size_of_cluster = len(cluster)
	for curr_data in cluster:
		for attr_ind in range(len(curr_data) - 1):
			if len(center) > attr_ind:
				center[attr_ind] += curr_data[attr_ind]
			else:
				center.append(curr_data[attr_ind])
	for attr_ind in range(len(center)):
		center[attr_ind] /= size_of_cluster
	return center

def get_centers(clusters: list)->list:
	centers = []
	for cluster in clusters:
		centers.append(get_center(cluster))
	return centers

def k_means(init_centers: list, list_of_data: list)->list:
	k = len(init_centers)
	centers = [centre for centre in init_centers]
	it = 0
	while True:
		it += 1
		clusters = get_curr_cluster(centers, list_of_data)
		new_centers = get_centers(clusters)		
		if new_centers == centers:
			break
		centers = [centre for centre in new_centers]
	print(centers)
	print(it)
	clusters = get_curr_cluster(centers, list_of_data)
	return clusters

def random_centers(k, list_of_data):
	init_centers = random.sample(list_of_data, k)
	for ind in range(k):
		init_centers[ind].pop()
	return init_centers

attributes, list_of_data = read_csv('diabetes.csv')
k = 10

init_centers = random_centers(k, list_of_data)
clusters = k_means(init_centers, list_of_data)
