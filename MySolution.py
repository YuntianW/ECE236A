import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
### TODO: import any other packages you need for your solution
from cvxopt import matrix, solvers

#--- Task 1 ---#
class MyClassifier:  
    def __init__(self, K):
        self.K = K  # number of classes

        ### TODO: Initialize other parameters needed in your algorithm
        # examples:
        # self.w = None
        # self.b = None

    
    def train(self, trainX):
        ''' Task 1-2 
            TODO: train classifier using LP(s) and updated parameters needed in your algorithm 
        '''
        
    
    def predict(self, testX):
        ''' Task 1-2 
            TODO: predict the class labels of input data (testX) using the trained classifier
        '''


        # Return the predicted class labels of the input data (testX)
        return predY
    

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)

        return accuracy
    

##########################################################################
#--- Task 2 ---#
class MyClustering:
    def __init__(self, K, beta=0.1, sigma=0.1):
        self.K = K  # number of classes
        self.labels = None

        # Sparse L1-Sparse Nonnegative Matrix Factorization
        # min |A-WH|_1 + beta*|H|_1 + sigma*|W|_infty
        self.beta = beta
        self.sigma = sigma
        self.W = None
        self.H = None
        self.loss = None
        
    
    def train(self, trainX, verbose=False):
        ''' Task 2-2 
            TODO: cluster trainX using LP(s) and store the parameters that discribe the identified clusters
        '''

        # intialize W and H
        A = trainX.T
        m, n = A.shape
        K = self.K
        W = np.random.rand(m, self.K)
        H = np.random.rand(self.K, n)

        # update W and H
        def update_H():
            new_H = np.zeros_like(H)

            c = matrix(np.concateate((self.beta * np.ones(K), np.ones(m))))
            A = matrix(np.block([
                    [-np.identity(K), np.zeros((K, m))],
                    [W, -np.identity(m)],
                    [W, -np.identity(m)]
                ]))
            for i in range(n):
                b = matrix(np.concatenate((np.zeros(self.K), -A[:, i], A[:, i])))
                sol = solvers.lp(c, A, b)
                new_H[:, i] = np.array(sol['x'])[:self.K]
            return new_H

        def update_W():
            new_W = np.zeros_like(W)

            c = matrix(np.concatenate((np.zeros(K), np.ones(n), self.gamma)))
            A = matrix(np.block([
                    [-H.T, -np.identity(n), -np.zeros((n, 1))],
                    [H.T, -np.identity(n), -np.zeros((n, 1))],
                    [np.identity(K), np.zeros((K, n)), -np.ones((K, 1))],
                    [-np.identity(K), np.zeros((K, n)), np.zeros((K, 1))]
                ]))
            for i in range(m):
                b = matrix(np.concatenate((-A[i, :], A[i, :], np.zeros(K), np.zeros(K))))
                sol = solvers.lp(c, A, b)
                new_W[i, :] = np.array(sol['x'])[:K]
            return new_W

        # update W and H iteratively
        for _ in range(100):
            H = update_H()
            W = update_W()
            self.loss = np.sum(np.abs(A - W @ H))
            if verbose:
                print('loss: ', self.loss)

        # Update and return the cluster labels of the training data (trainX)
        self.W = W
        self.H = H
        self.labels = np.argmax(H, axis=0)

        return self.labels


    def infer_cluster(self, testX):
        ''' Task 2-2 
            TODO: assign new data points to the existing clusters
        '''

        # Return the cluster labels of the input data (testX)
        return pred_labels
    

    def evaluate_clustering(self,trainY):
        label_reference = self.align_cluster_labels(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(self.labels, label_reference)
        nmi = normalized_mutual_info_score(trainY, aligned_labels)

        return nmi
    

    def evaluate_classification(self, trainY, testX, testY):
        pred_labels = self.infer_data_labels(testX)
        label_reference = self.align_cluster_labels(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(pred_labels, label_reference)
        accuracy = accuracy_score(testY, aligned_labels)

        return accuracy


    def get_class_cluster_reference(cluster_labels, true_labels):
        ''' assign a class label to each cluster using majority vote '''
        label_reference = {}
        for i in range(len(np.unique(cluster_labels))):
            index = np.where(cluster_labels == i,1,0)
            num = np.bincount(true_labels[index==1]).argmax()
            label_reference[i] = num

        return label_reference
    
    
    def align_cluster_labels(cluster_labels, reference):
        ''' update the cluster labels to match the class labels'''
        aligned_lables = np.zeros_like(cluster_labels)
        for i in range(len(cluster_labels)):
            aligned_lables[i] = reference[cluster_labels[i]]

        return aligned_lables



##########################################################################
#--- Task 3 ---#
class MyLabelSelection:
    def __init__(self, ratio):
        self.ratio = ratio  # percentage of data to label
        ### TODO: Initialize other parameters needed in your algorithm

    def select(self, trainX):
        ''' Task 3-2'''
        

        # Return an index list that specifies which data points to label
        return data_to_label

    