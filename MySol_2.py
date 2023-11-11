from utils import *
import matplotlib.pyplot as plt
import sys
# from MySolution import MyClassifier, MyClustering, MyLabelSelection
import utils
from scipy.optimize import linprog

mnist_data = prepare_mnist_data()
# mnist_data=prepare_synthetic_data()
print("Synthetic data shape: ", mnist_data['trainX'].shape, mnist_data['trainY'].shape)
class_type = set(mnist_data['trainY'])
class_type = np.array(list(class_type), dtype=np.int)
trainY = []
trainX = []
testX = mnist_data['testX']
testY = mnist_data['testY']
count = 0
for i in class_type:
    trainY.append(np.where(mnist_data['trainY'] == i)[0])
    # testY.append(np.where(mnist_data['testY'] == i)[0])
    trainX.append(mnist_data['trainX'][trainY[count], :])
    # testX.append(mnist_data['testX'][testY[count], :])
    count += 1

class_1 = trainX[0]
class_2 = trainX[1]


# epsilon = -0.1
'''
we want to min(max(0,y-(ax+b+/-epsilon)))
it means whether a data is clear seperated or there exists a penalty(it means the larger one compare to 0)
'''
def train_two_class(class_1, class_2, epsilon):
    A = np.zeros((2 * (len(class_1) + len(class_2)), class_1.shape[1] + 1 + len(class_1) + len(class_2)))
    b = np.zeros((2 * (len(class_1) + len(class_2)), 1))
    c = np.zeros((1, class_1.shape[1] + 1 + len(class_1) + len(class_2)))
    A[:len(class_1), :class_1.shape[1]] = class_1[:, :]
    A[:len(class_1), class_1.shape[1]] = 1
    A[len(class_1):len(class_1) + len(class_2), :class_1.shape[1]] = -class_2[:, :]
    A[len(class_1):len(class_1) + len(class_2), class_1.shape[1]] = 1
    A[0:len(class_1) + len(class_2), (class_1.shape[1] + 1):] = -np.diag(np.ones(len(class_1) + len(class_2)))
    A[len(class_1) + len(class_2):, (class_1.shape[1] + 1):] = -np.diag(np.ones(len(class_1) + len(class_2)))


    b[:len(class_1)] = 1 + epsilon
    b[len(class_1):len(class_1) + len(class_2)] = -1 + epsilon

    c[0, (class_1.shape[1] + 1):] = np.ones((1, len(class_1) + len(class_2)))
    res = linprog(c=c, A_ub=A, b_ub=b)
    a_res = res.x[:class_1.shape[1]]
    b_res = res.x[class_1.shape[1]]
    if np.mean(class_1 @ a_res + b_res) < np.mean(class_2 @ a_res + b_res):
        which_one_larger = 1
        acc1 = np.sum(class_1 @ a_res + b_res < 1) / class_1[:, 0].shape[0]
        acc2 = np.sum(class_2 @ a_res + b_res > 1) / class_2[:, 0].shape[0]
    else:
        which_one_larger = 0
        acc1 = np.sum(class_1 @ a_res + b_res > 1) / class_1[:, 0].shape[0]
        acc2 = np.sum(class_2 @ a_res + b_res < 1) / class_2[:, 0].shape[0]
    print(acc1, acc2)
    return res.x[:class_1.shape[1] + 1], which_one_larger


data = trainX
epsilon = -0.1

'''
In train_two_class(), it will return the solution of the equation and which class is larger
So in train_all_classes(), we will use the solution to train all the classes
for each class, we will compare it with other classes by the solution we get from train_two_class()
'''
def train_all_classes(data, epsilon):
    order = np.zeros((int(len(data) * (len(data) - 1) / 2), 2), dtype=np.int)
    sol = np.zeros((int(len(data) * (len(data) - 1) / 2), data[0].shape[1] + 1))
    count = 0
    for i in range(0, len(data)):
        for j in range(i + 1, len(data)):
            order[count, 0] = i
            order[count, 1] = j
            count += 1
    matrix = np.zeros((len(data), int(len(data) * (len(data) - 1) / 2)))
    for i in range(0, order.shape[0]):
        sol_temp, which_larger = train_two_class(data[order[i][0]], data[order[i][1]], epsilon)
        sol[i, :] = (sol_temp)
        matrix[order[i][0], i] = 1 if which_larger == 0 else -1
        matrix[order[i][1], i] = 1 if which_larger == 1 else -1
    return sol, matrix


solution = train_all_classes(data, epsilon)


def test_classify_class(data_x, data_y, solution):
    class_type = set(data_y)
    class_type = np.array(list(class_type), dtype=np.int)
    result = np.zeros((1, len(data_x)))
    sol = solution[0]
    matrix = solution[1]
    for i in range(0, len(data_x)):  # check each data
        class_temp = np.zeros((matrix.shape[0], 1))
        for j in range(0, matrix.shape[0]):  # check which class
            temp = True
            for k in range(0, matrix.shape[1]):  # check the equations
                if matrix[j, k] != 0:
                    temp = temp & (
                            (data_x[i, :] @ sol[k, :-1] + sol[k, -1] - 1) * matrix[j, k] > 0)
            if temp:
                class_temp[j, 0] = 1
        result[0, i] = class_type[np.argmax(class_temp)]
    acc = np.sum(result == data_y) / len(data_y)
    return result, acc


predict, accuracy = test_classify_class(testX, testY, solution)
print(accuracy)
