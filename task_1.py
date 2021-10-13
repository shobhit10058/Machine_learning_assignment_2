import k_means

attributes, list_of_data = k_means.read_csv('mean_data.csv')
k_means.normalize_data(list_of_data)
print("Enter the value of k: ", end=' ')
k = int(input())

init_centers = k_means.random_centers(k, list_of_data)
clusters = k_means.k_means(init_centers, list_of_data)