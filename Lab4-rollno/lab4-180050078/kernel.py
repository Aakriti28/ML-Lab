import numpy as np 

def linear_kernel(X,Y,sigma=None):
	'''Returns the gram matrix for a linear kernel
	
	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma - dummy argment, don't use
	Return:
		K - numpy array of size n x m
	''' 
	# TODO 
	return X @ Y.T
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE
	pass
	# END TODO

def gaussian_kernel(X,Y,sigma=0.1):
	'''Returns the gram matrix for a rbf
	
	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma - The sigma value for kernel
	Return:
		K - numpy array of size n x m
	'''
	# TODO
	s=np.shape(X)
	d=s[1]
	n=s[0]
	s2=np.shape(Y)
	m=s2[0]
	rown=np.sum(X**2,axis=1).reshape((n,1))
	coln=np.sum(Y**2,axis=1).reshape((1,m))
	K=-2*(X @ Y.T) 
	K=K+rown+coln
	return np.exp(-1*K/(2*sigma*sigma))
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE 
	pass
	# END TODO

def my_kernel(X,Y,sigma):
	'''Returns the gram matrix for your designed kernel
	
	Arguments:
		X - numpy array of size n x d
		Y - numpy array of size m x d
		sigma- dummy argment, don't use
	Return:
		K - numpy array of size n x m
	''' 
	# TODO
	# NOTE THAT YOU CANNOT USE FOR LOOPS HERE 
	return np.power((1+ (X @ Y.T)),4)
	pass
	# END TODO
