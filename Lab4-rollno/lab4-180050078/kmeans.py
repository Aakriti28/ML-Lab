import argparse
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(5927)

class KMeans():
	def __init__(self, D, n_clusters):
		self.n_clusters = n_clusters
		self.cluster_centers = np.zeros((n_clusters, D))

	def init_clusters(self, data):
		### TODO
		### Initialize cluster_centers using n_clusters points sampled from data
		lis=np.random.choice(data.shape[0],self.n_clusters,replace=False)
		self.cluster_centers=data[lis,:]


		### END TODO

	def pred(self, x):
		### TODO: Given a sample x, return id of closest cluster center
		# diff=(self.cluster_centers-x)**2
		# print(diff.shape)
		return np.argmin(np.sum((self.cluster_centers-x)**2,axis=1))
		### END TODO

	def train(self, data, max_iter=10000, epsilon=1e-4):
		for it in range(max_iter):
			### TODO
			### Declare and initialize required variables
			n=data.shape[0]
			lb=np.zeros((n,1))
			new_cen=np.zeros(self.cluster_centers.shape)
			new_cen[:,:]=self.cluster_centers[:,:]
			# print(self.cluster_centers)
			### Update labels for each point
			centres_ex = self.cluster_centers[: , np.newaxis, :]
			lb=np.argmin(np.sum(((data - centres_ex)**2),axis=2), axis=0)
			# print(lb.shape)

			# for i, x in enumerate(data):
			# 	lb[i] = self.pred(x)
			# print(lb)
			# print(data.shape)
			### Update cluster centers
			### Note: If some cluster is empty, do not update the cluster center
			for i in range(self.n_clusters): 
				lis=(lb==i)
				# print(lis.shape)
				if(np.sum(lis)==0):
					continue
				# print(lis.shape)
				# print(lb.shape)
				# print(data[lis,:].shape)
				# f=np.mean(data[lis,:],axis=1)
				# print(f.shape)
				new_cen[i,:]=np.mean(data[lis,:],axis=0)


			### Check for convergence
			### Stop if distance between each of the old and new cluster centers is less than epsilon
			if(np.max(np.sqrt(np.sum((new_cen-self.cluster_centers)**2,axis=1)))<epsilon):
				break
			self.cluster_centers[:,:]=new_cen[:,:]
			### END TODO
		return it

	def replace_by_center(self, data):
		out = np.zeros_like(data)
		for i, x in enumerate(data):
			out[i] = self.cluster_centers[self.pred(x)]
		return out

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--image', default='1', choices=['1', '2', '3'])
	parser.add_argument('--k', default=5, type=int)

	args = parser.parse_args()

	image = plt.imread(f'data/{args.image}.png')
	x = image.reshape(-1, 3)
	kmeans = KMeans(D=3, n_clusters=args.k)
	kmeans.init_clusters(x)
	kmeans.train(x)
	out = kmeans.replace_by_center(x)
	plt.imshow(out.reshape(image.shape))
	plt.show()
