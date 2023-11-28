import numpy as np
from sklearn.metrics import normalized_mutual_info_score, accuracy_score
### TODO: import any other packages you need for your solution
from scipy.optimize import linprog
from scipy.spatial.distance import cosine
from tqdm import tqdm
import random

#--- Task 1 ---#
class MyClassifier:
    def __init__(self, K, epsilon=0.1):
        self.K = K 
        self.epsilon = epsilon
        self.solution = None
        self.matrix = None

    def train_two_class(self, class_1, class_2):
        data_0, data_1 = (class_1, class_2) if np.mean(class_1[:, 1]) > np.mean(class_2[:, 1]) else (class_2, class_1)

        A = np.zeros((2 * (len(data_0) + len(data_1)), 2 + len(data_0) + len(data_1)))
        b = np.zeros((2 * (len(data_0) + len(data_1)), 1))

        A[:len(data_0), 0] = data_0[:, 0]
        A[:len(data_0), 1] = 1
        b[:len(data_0),0] = data_0[:, 1] - self.epsilon

        A[len(data_0):len(data_0) + len(data_1), 0] = -data_1[:, 0]
        A[len(data_0):len(data_0) + len(data_1), 1] = -1
        b[len(data_0):len(data_0) + len(data_1),0] = -data_1[:, 1] - self.epsilon
        
        A[:len(data_0) + len(data_1), 2:] = -np.eye(len(data_0) + len(data_1))
        A[len(data_0) + len(data_1):, 2:] = -np.eye(len(data_0) + len(data_1))

        c = np.ones(len(data_0) + len(data_1) + 2)
        c[:2] = 0 
        res = linprog(c=c, A_ub=A, b_ub=b)
        sol = res.x[:2]
        return sol

    def train_all_classes(self, data_class):
        num_combinations = self.K * (self.K - 1) // 2
        sol = np.zeros((num_combinations, 2))
        matrix = np.zeros((self.K, num_combinations))
        count = 0

        for i in range(self.K):
            for j in range(i + 1, self.K):
                sol_temp = self.train_two_class(data_class[i], data_class[j])
                sol[count] = sol_temp

                matrix[i, count] = 1 if np.mean(data_class[i][:, 1]) > np.mean(data_class[j][:, 1]) else -1
                matrix[j, count] = 1 if np.mean(data_class[i][:, 1]) < np.mean(data_class[j][:, 1]) else -1
                count += 1

        return sol, matrix

    def train(self, data):
        index_class = [np.where(data['trainY'] == k)[0] for k in range(self.K)]
        data_class = [data['trainX'][indices] for indices in index_class]
        self.solution, self.matrix = self.train_all_classes(data_class)

    def predict(self, testX):
        result = np.zeros(len(testX))
        for i in range(len(testX)):
            class_temp = np.zeros(self.K)
            for j in range(self.K):
                temp = True
                for k in range(len(self.solution)):
                    if self.matrix[j, k] != 0:
                        temp = temp and ((testX[i, 1] - (testX[i, 0] * self.solution[k, 0] + self.solution[k, 1])) * self.matrix[j, k] > 0)
                if temp:
                    class_temp[j] = 1
            result[i] = np.argmax(class_temp)
        return result

    def evaluate(self, testX, testY):
        predY = self.predict(testX)
        accuracy = accuracy_score(testY, predY)
        return accuracy

##########################################################################
#--- Task 2 ---#

def solve_l1_Ax_b(A, b):
    '''
        min ||Ax-b||_1
        s.t x>=0
            1^Tx=1
            x integer
    '''
    m, n = A.shape
    c = np.concatenate((np.zeros(n), np.ones(m)))
    A_ub = np.block([
        [-np.identity(n), np.zeros((n, m))], 
        [A, -np.identity(m)],
        [-A, -np.identity(m)]
    ])
    A_eq = np.block(
        [np.ones((1, n)), np.zeros((1, m))]
    )
    b_ub = np.concatenate((np.zeros(n), b, -b))
    b_eq = 1
    integrality = np.concatenate((np.ones(n), np.zeros(m)))
    sol = linprog(c=c, A_eq=A_eq, b_eq=b_eq, A_ub=A_ub, b_ub=b_ub, integrality=integrality)
    return sol["x"][: n]


def solve_l1_l1_Ax_b(A, b, beta):
    '''
        min ||Ax - b||_1 + beta * ||x||_1
        s.t x>=0
    '''
    m, n = A.shape
    c = np.concatenate((np.zeros(n), np.ones(m), beta*np.ones(n)))
    A_ub = np.block([
        [A, -np.identity(m), np.zeros((m, n))],
        [-A, -np.identity(m), np.zeros((m, n))],
        [-np.identity(n), np.zeros((n, m)), np.zeros((n, n))],
        [np.identity(n), np.zeros((n, m)), -np.identity(n)]
    ])
    b_ub = np.concatenate((b, -b, np.zeros(n), np.zeros(n)))
    sol = linprog(c=c, A_ub=A_ub, b_ub=b_ub)
    return sol['x'][: n]


def solve_l1_linf_Ax_b(A, b, beta):
    '''
        min ||Ax - b||_1 + beta * ||x||_inf
        s.t x>=0
    '''
    m, n = A.shape
    c = np.concatenate((np.zeros(n), np.ones(m), beta*np.ones(1)))
    A_ub = np.block([
        [A, -np.identity(m), np.zeros((m, 1))],
        [-A, -np.identity(m), np.zeros((m, 1))],
        [-np.identity(n), np.zeros((n, m)), np.zeros((n, 1))],
        [np.identity(n), np.zeros((n, m)), -np.ones((n, 1))]
    ])
    b_ub = np.concatenate((b, -b, np.zeros(n), np.zeros(n)))
    sol = linprog(c=c, A_ub=A_ub, b_ub=b_ub)
    return sol['x'][: n]


class MyClustering:
    def __init__(self, K):
        '''
            Sparse Nonnegative Matrix Factorization
            min |A-WH|_1 + beta*sum|H_i|_1 + gamma*|W|_inf
            https://faculty.cc.gatech.edu/~hpark/papers/GT-CSE-08-01.pdf
        '''
        self.K = K  # number of classes
        self.labels = None

        self.W = None
        self.H = None
        self.loss = None
    
    def train(self, trainX, beta=0.1, gamma=0.5, maxiter=5, verbose=False):
        ''' Task 2-2 
            TODO: cluster trainX using LP(s) and store the parameters that discribe the identified clusters
        '''

        # intialize W and H
        A = trainX.T
        W = A[:, np.random.randint(0, A.shape[1], self.K)]
        H = np.zeros((self.K, A.shape[1]))

        # update W and H iteratively
        for _ in tqdm(range(maxiter)):
            for i in range(A.shape[1]):
                H[:, i] = solve_l1_l1_Ax_b(W, A[:, i], gamma)
            for i in range(A.shape[0]):
                W[i, :] = solve_l1_linf_Ax_b(H.T, A[i, :], beta)
            self.loss = np.sum(np.abs(A-W@H)) / (A.shape[0]*A.shape[1])
            if verbose:
                print('loss: ', self.loss)

        # Update and return the cluster labels of the training data (trainX)
        self.W = W
        self.H = H
        self.labels = H.argmax(axis=0)

        return self.labels


    def infer_cluster(self, testX):
        ''' Task 2-2 
            TODO: assign new data points to the existing clusters
        '''

        # Return the cluster labels of the input data (testX)
        A = testX.T
        H = np.zeros((self.K, A.shape[1]))
        for i in range(A.shape[1]):
            H[:, i] = solve_l1_Ax_b(self.W, A[:, i])

        pred_labels = H.argmax(axis=0)
        return pred_labels
    

    def evaluate_clustering(self,trainY):
        label_reference = self.align_cluster_labels(self.labels, trainY)
        aligned_labels = self.align_cluster_labels(self.labels, label_reference)
        nmi = normalized_mutual_info_score(trainY, aligned_labels)

        return nmi


    def evaluate_classification(self, trainY, testX, testY):
        pred_labels = self.infer_cluster(testX)
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
    
    
    def align_cluster_labels(self, cluster_labels, reference):
        ''' update the cluster labels to match the class labels'''
        aligned_lables = np.zeros_like(cluster_labels)
        for i in range(len(cluster_labels)):
            aligned_lables[i] = reference[cluster_labels[i]]

        return aligned_lables



##########################################################################
#--- Task 3 ---#


def min_l1_Ax(A):
    m, n = A.shape
    c = np.ones((1, m)) @ A
    A_eq = np.ones((1, n))
    b_eq = 1
    integrality = np.ones(n)

    sol = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=(0, None), integrality=integrality, method='interior-point')
    return sol['x']

def min_l1_Ax_relaxed(A):
    m, n = A.shape
    c = np.ones((1, m)) @ A
    A_eq = np.ones((1, n))
    b_eq = 1

    # Remove the integrality constraint
    sol = linprog(c=c, A_eq=A_eq, b_eq=b_eq, bounds=(0, 1.1), method='highs')
    return sol['x']

class MyLabelSelection:
    def __init__(self, ratio):
        self.ratio = ratio  # percentage of data to label
        ### TODO: Initialize other parameters needed in your algorithm
        self.W = None

    def select(self, trainX):
        copy=trainX.copy()
        A = copy.T
        A /= np.linalg.norm(A, axis=0)
        m, n = A.shape
        W = np.zeros((1, m))
        labels = list(range(n))
        mean_sample = np.mean(trainX, axis=0)
        distances = np.linalg.norm(trainX - mean_sample, axis=1)
        #sel = np.argmax(distances)
        sel = np.random.randint(n)
        selected = set()
        selected.add(sel)
        print(selected)
        W[0, :] = A[:, sel]
        labels.pop(sel)
        A = np.delete(A, sel, axis=1)

        for i in tqdm(range(int(n*self.ratio)-1)):
            sel = min_l1_Ax_relaxed(W@A).argmax()
            selected.add(labels[sel])
            W = np.vstack((W, A[:, sel]))
            A = np.delete(A, sel, axis=1)
            labels.pop(sel)


        self.W = W
        # Return an index list that specifies which data points to label
        return list(selected)

