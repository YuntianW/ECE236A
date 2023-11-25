from utils import *
import matplotlib.pyplot as plt
import sys
from MySolution import MyClassifier, MyClustering, MyLabelSelection
import utils
from scipy.optimize import linprog
from scipy.spatial.distance import cdist

class random_selector:
    def __init__(self, ratio):
        self.ratio = ratio 
    def select(self, trainX):
        total_samples = len(trainX)
        sample_size = int(self.ratio * total_samples)
        selected_indices = np.random.choice(total_samples, size=sample_size, replace=False)
        print(selected_indices)

        return selected_indices


syn_data = prepare_synthetic_data()
mnist_data=prepare_mnist_data()
print("Synthetic data shape: ", syn_data['trainX'].shape, syn_data['trainY'].shape)
test_x = syn_data['testX']
test_y = syn_data['testY']

#label_selector = random_selector(ratio=0.04)
label_selector = MyLabelSelection(ratio=0.05)
index_class = [np.where(syn_data['trainY'] == i)[0] for i in range(3)]
selected_indices_class = []

for indices in index_class:
    class_data = syn_data['trainX'][indices, :]
    selected_indices = label_selector.select(class_data)
    selected_indices_class.append(indices[selected_indices])

data_class = [syn_data['trainX'][selected_indices, :] for selected_indices in selected_indices_class]

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


solution = train_all_classes(data_class, 0.1)


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
plt.figure()
plt.subplot(1, 2, 1)
plt.scatter(test_x[:, 0], test_x[:, 1], c=result)
plt.title('My result, Accuracy=%.3f'%(acc))
plt.subplot(1, 2, 2)
plt.scatter(test_x[:, 0], test_x[:, 1], c=test_y)
plt.title('Ground Truth')
plt.show()
