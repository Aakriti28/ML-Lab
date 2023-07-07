import numpy as np
from kernel import *
from utils import *
import matplotlib.pyplot as plt


class KernelLogistic(object):
    def __init__(self, kernel=gaussian_kernel, iterations=100,eta=0.01,lamda=0.05,sigma=1):
        self.kernel = lambda x,y: kernel(x,y,sigma)
        self.iterations = iterations
        self.alpha = None
        self.eta = eta     # Step size for gradient descent
        self.lamda = lamda # Regularization term

    def fit(self, X, y):
        ''' find the alpha values here'''
        self.train_X = X
        self.train_y = y
        self.alpha = np.zeros((y.shape[0],1))
        kernel = self.kernel(self.train_X,self.train_X)

        # TODO
        for _ in range(self.iterations):
            # TODO: Update the weights using a single step of gradient descent. You are not allowed to use loops here.
            # if(iter%1000==0):
            #     print(self.weights)
            sig=1/(1+np.exp(-1*(kernel @ self.alpha)))
            grad=kernel @ (self.train_y[:,None] - sig - self.lamda*self.alpha)
            # print(grad.shape)
            self.alpha=self.alpha+self.eta*grad

            # print(self.alpha.shape)
            # print(y.shape)
            # print(np.linalg.norm(grad))
            # END TODO

            # TODO: Stop the algorithm if the norm of the gradient falls below 1e-4
            # if(np.linalg.norm(grad)<1e-4):
            #     # print("converged")
            #     break

        # END TODO
    

    def predict(self, X):
        # TODO 
        kernel = self.kernel(X,self.train_X)
        # print(kernel.shape)
        # wt=np.diag(self.alpha[:,0])
        # # print(wt.shape)
        # wt_ker=kernel @ wt
        tmp= 1/(1+np.exp(-1*(kernel @ self.alpha)))
        return tmp[:,0]
        # END TODO

def k_fold_cv(X,y,k=10,sigma=1.0):
    '''Does k-fold cross validation given train set (X, y)
    Divide train set into k subsets, and train on (k-1) while testing on 1. 
    Do this process k times.
    Do Not randomize 
    
    Arguments:
        X  -- Train set
        y  -- Train set labels
    
    Keyword Arguments:
        k {number} -- k for the evaluation
        sigma {number} -- parameter for gaussian kernel
    
    Returns:
        error -- (sum of total mistakes for each fold)/(k)
    '''
    # TODO 
    sig=sigma
    n=X.shape[0]
    # print(X.shape,y.shape)
    sz=n//k
    tot=0
    for i in range(k):
        st=i*sz
        end=(i+1)*sz
        if(i==k):
            end=n
        ms=np.ones((n,),dtype=bool)
        ms[st:end]=0
        clf = KernelLogistic(gaussian_kernel,sigma=sig)
        clf.fit(X[ms,:], y[ms])

        y_predict = clf.predict(X[st:end]) > 0.5

        err = np.sum(y_predict != y[st:end])
        tot+=err
        # print(i)

    return tot/k



    # END TODO

if __name__ == '__main__':
    data = np.loadtxt("./data/dataset1.txt")
    X1 = data[:900,:2]
    Y1 = data[:900,2]

    clf = KernelLogistic(gaussian_kernel)
    clf.fit(X1, Y1)

    y_predict = clf.predict(data[900:,:2]) > 0.5

    correct = np.sum(y_predict == data[900:,2])
    print("%d out of %d predictions correct" % (correct, len(y_predict)))
    if correct > 92:
        marks = 1.0
    else:
        marks = 0
    print(f"You recieve {marks} for the fit function")

    errs = []
    sigmas = [0.5, 1, 2, 3, 4, 5, 6]
    for s in sigmas:  
      errs+=[(k_fold_cv(X1,Y1,sigma=s))]
    plt.plot(sigmas,errs)
    plt.xlabel('Sigma')
    plt.ylabel('Mistakes')
    plt.title('A plot of sigma v/s mistakes')
    plt.show()
