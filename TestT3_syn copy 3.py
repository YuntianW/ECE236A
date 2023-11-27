from utils import *
import matplotlib.pyplot as plt
import sys
from MySolution import MyClassifier, MyClustering, MyLabelSelection
import utils
from scipy.optimize import linprog
from scipy.spatial.distance import cdist
from sklearn.svm import SVC
from sklearn.svm import SVC
import numpy as np
import time
start_time = time.time() 

class CustomSampleSelector:
    def __init__(self, initial_samples_per_class, target_size_per_class,epsilon=0.5):
        self.initial_samples_per_class = initial_samples_per_class
        self.target_size_per_class = target_size_per_class
        self.epsilon = epsilon

    @staticmethod
    def _train_lp_find_hyperplane(class_1, class_2, epsilon):
        if np.mean(class_1[:, 1]) > np.mean(class_2[:, 1]):
            data_0 = class_1
            data_1 = class_2
        else:
            data_0 = class_2
            data_1 = class_1
        A = np.zeros((2 * (len(data_0) + len(data_1)), 2 + len(data_0) + len(data_1)))
        b = np.zeros((2 * (len(data_0) + len(data_1)), 1))

        A[:len(data_0), 0] = data_0[:, 0]
        A[:len(data_0), 1] = 1

        A[len(data_0):len(data_0) + len(data_1), 0] = -data_1[:, 0]
        A[len(data_0):len(data_0) + len(data_1), 1] = -1

        A[0:len(data_0) + len(data_1), 2:] = -np.diag(np.ones(len(data_0) + len(data_1)))
        A[len(data_0) + len(data_1):, 2:] = -np.diag(np.ones(len(data_0) + len(data_1)))

        b[:len(data_0), 0] = data_0[:, 1] - epsilon
        b[len(data_0):len(data_0) + len(data_1), 0] = -data_1[:, 1] - epsilon

        c = np.zeros((1, len(data_0) + len(data_1) + 2))
        c[0, 2:] = np.ones((1, len(data_0) + len(data_1)))
        res = linprog(c=c, A_ub=A, b_ub=b)
        sol = res['x'][0:2]
        accuracy_0 = np.sum(data_0[:, 1] > (sol[0] * data_0[:, 0] + sol[1])) / data_0[:, 0].shape[0]
        accuracy_1 = np.sum(data_1[:, 1] < (sol[0] * data_1[:, 0] + sol[1])) / data_1[:, 0].shape[0]
        return sol

    def select_samples(self, data, labels):
        num_classes = len(set(labels))
        selected_indices_by_class = {i: set() for i in range(num_classes)}

        # Initial random selection for each class
        for class_index in range(num_classes):
            initial_indices = np.random.choice(np.where(labels == class_index)[0], self.initial_samples_per_class, replace=False)
            selected_indices_by_class[class_index].update(initial_indices)

        for i in range(num_classes):
            for j in range(i + 1, num_classes):
                # Iterate to select new samples
                for _ in range(self.target_size_per_class):
                    # Recalculate hyperplane with current selected samples
                    indices_i = list(selected_indices_by_class[i])
                    indices_j = list(selected_indices_by_class[j])
                    hyperplane = self._train_lp_find_hyperplane(data[indices_i], data[indices_j], self.epsilon)
                    print(hyperplane)

                    # Select one more sample for class i and update hyperplane if a new sample is added
                    sample_added_i = False
                    for index in np.where(labels == i)[0]:
                        if index not in selected_indices_by_class[i]:
                            distance = np.dot(data[index], hyperplane)
                            if abs(distance) <= self.epsilon:
                                selected_indices_by_class[i].add(index)
                                sample_added_i = True
                                break


                    # Select one more sample for class j and update hyperplane if a new sample is added
                    sample_added_j = False
                    for index in np.where(labels == j)[0]:
                        if index not in selected_indices_by_class[j]:
                            distance = np.dot(data[index], hyperplane)
                            if abs(distance) <= self.epsilon:
                                selected_indices_by_class[j].add(index)
                                sample_added_j = True
                                break

        # Compile the final selected data for each class
        data_class = []
        for class_index in range(num_classes):
            selected_indices = list(selected_indices_by_class[class_index])
            class_data = data[selected_indices]
            data_class.append(class_data)

        return data_class

syn_data = prepare_synthetic_data()
#mnist_data=prepare_mnist_data()
print("Synthetic data shape: ", syn_data['trainX'].shape, syn_data['trainY'].shape)
test_x = syn_data['testX']
test_y = syn_data['testY']
index_class = [np.where(syn_data['trainY'] == i)[0] for i in range(3)]

selector = CustomSampleSelector(initial_samples_per_class=15, target_size_per_class=8, epsilon=0.5)
data_class = selector.select_samples(syn_data['trainX'], syn_data['trainY'])

epsilon = 3
index0 = 0
index1 = 1
index2 = 2
data_0 = data_class[index0]
data_1 = data_class[index1]
data_2 = data_class[index2]

label_0 = index_class[index0]
label_1 = index_class[index1]
label_2 = index_class[index2]


def train_two_class(class_1, class_2, epsilon=.1):
    if np.mean(class_1[:, 1]) > np.mean(class_2[:, 1]):
        data_0 = class_1
        data_1 = class_2
    else:
        data_0 = class_2
        data_1 = class_1
    A = np.zeros((2 * (len(data_0) + len(data_1)), 2 + len(data_0) + len(data_1)))
    b = np.zeros((2 * (len(data_0) + len(data_1)), 1))

    A[:len(data_0), 0] = data_0[:, 0]
    A[:len(data_0), 1] = 1

    A[len(data_0):len(data_0) + len(data_1), 0] = -data_1[:, 0]
    A[len(data_0):len(data_0) + len(data_1), 1] = -1

    A[0:len(data_0) + len(data_1), 2:] = -np.diag(np.ones(len(data_0) + len(data_1)))
    A[len(data_0) + len(data_1):, 2:] = -np.diag(np.ones(len(data_0) + len(data_1)))

    b[:len(data_0), 0] = data_0[:, 1] - epsilon
    b[len(data_0):len(data_0) + len(data_1), 0] = -data_1[:, 1] - epsilon

    c = np.zeros((1, len(data_0) + len(data_1) + 2))
    c[0, 2:] = np.ones((1, len(data_0) + len(data_1)))
    res = linprog(c=c, A_ub=A, b_ub=b)
    sol = res['x'][0:2]
    accuracy_0 = np.sum(data_0[:, 1] > (sol[0] * data_0[:, 0] + sol[1])) / data_0[:, 0].shape[0]
    accuracy_1 = np.sum(data_1[:, 1] < (sol[0] * data_1[:, 0] + sol[1])) / data_1[:, 0].shape[0]
    return sol, accuracy_0, accuracy_1


def train_all_classes(data, epsilon):
    order = np.zeros((int(len(data) * (len(data) - 1) / 2), 2), dtype=np)
    sol = np.zeros((int(len(data) * (len(data) - 1) / 2), 2))
    count = 0
    for i in range(0, len(data)):
        for j in range(i + 1, len(data)):
            order[count, 0] = i
            order[count, 1] = j
            count += 1

    matrix = np.zeros((len(data), int(len(data) * (len(data) - 1) / 2)))
    for i in range(0, order.shape[0]):
        sol_temp, _, _ = train_two_class(data_class[order[i][0]], data_class[order[i][1]], epsilon)
        sol[i, :] = (sol_temp)
        matrix[order[i][0], i] = 1 if np.mean(data_class[order[i][0]][:, 1]) > np.mean(
            data_class[order[i][1]][:, 1]) else -1
        matrix[order[i][1], i] = 1 if np.mean(data_class[order[i][0]][:, 1]) < np.mean(
            data_class[order[i][1]][:, 1]) else -1
    return sol, matrix


solution = train_all_classes(data_class, 0.02)


def test_classify_class(data_x, data_y, solution):
    result = np.zeros((1, len(data_x)))
    for i in range(0, len(data_x)):
        class_temp = np.zeros((solution[1].shape[0], 1))
        for j in range(0, solution[1].shape[0]):
            temp = True
            for k in range(0, solution[1].shape[1]):
                if solution[1][j, k] != 0:
                    temp = temp & (
                                (data_x[i, 1] - (data_x[i, 0] * solution[0][k, 0] + solution[0][k, 1])) * solution[1][
                            j, k] > 0)
            if temp:
                class_temp[j, 0] = 1
        result[0, i] = np.argmax(class_temp)

    acc = np.sum(result == data_y) / len(data_y)
    return result, acc

result, acc = test_classify_class(data_x=test_x, data_y=test_y, solution=solution)

end_time = time.time()
elapsed_time = end_time - start_time

print(f"Finding the hyperplane took {elapsed_time} seconds")

plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(test_x[:, 0], test_x[:, 1], c=result)
plt.title('My result, Accuracy=%.3f'%(acc))
plt.subplot(1, 2, 2)
plt.scatter(test_x[:, 0], test_x[:, 1], c=test_y)
plt.title('Ground Truth')
plt.show()

