from utils import *
import matplotlib.pyplot as plt
import sys
from MySolution import MyClassifier, MyClustering, MyLabelSelection
import utils
from scipy.optimize import linprog


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


'''
In train_two_class(), it will return the solution of the equation and which class is larger
So in train_all_classes(), we will use the solution to train all the classes
for each class, we will compare it with other classes by the solution we get from train_two_class()
'''
def train_all_classes(data, epsilon):
    order = np.zeros((int(len(data) * (len(data) - 1) / 2), 2), dtype=np)
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


def test_classify_class(data_x, data_y, solution):
    class_type = set(data_y)
    class_type = np.array(list(class_type), dtype=np)
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

class random_selector:
    def __init__(self, ratio):
        self.ratio = ratio  
    def select(self, trainX):
        total_samples = len(trainX)
        sample_size = int(self.ratio * total_samples)
        selected_indices = np.random.choice(total_samples, size=sample_size, replace=False)

        return selected_indices

label_percentages = [0.05, 0.1, 0.2, 0.5]
accuracies_our_algo = []
accuracies_random = []

random_selector = random_selector(ratio=None)
label_selector = MyLabelSelection(ratio=None)

for ratio in label_percentages:

    mnist_data = prepare_mnist_data()
    print("MNIST data shape: ", mnist_data['trainX'].shape, mnist_data['trainY'].shape)

    label_selector.ratio = ratio
    random_selector.ratio = ratio
    selected_indices=[]
    selected_samples=[]
    selected_labels=[]

    selected_indices = label_selector.select(mnist_data['trainX'])
    selected_samples = mnist_data['trainX'][selected_indices]
    selected_labels = mnist_data['trainY'][selected_indices]


    class_type = set(mnist_data['trainY'])
    class_type = np.array(list(class_type), dtype=np)
    trainY = []
    trainX = []

    count = 0
    for i in class_type:
        trainY.append(np.where(selected_labels == i)[0])
        # testY.append(np.where(mnist_data['testY'] == i)[0])
        trainX.append(selected_samples[trainY[count], :])
        # testX.append(mnist_data['testX'][testY[count], :])
        count += 1


    data=trainX

    class_1 = trainX[0]
    class_2 = trainX[1]

    epsilon = -0.5
    solution = train_all_classes(data, epsilon)

    testX = mnist_data['testX']
    testY = mnist_data['testY']

    predict, accuracy = test_classify_class(testX, testY, solution)
    print("Test Accuracy: ", accuracy)

    accuracies_our_algo.append(accuracy)

    selected_indices = random_selector.select(mnist_data['trainX'])
    selected_samples = mnist_data['trainX'][selected_indices]
    selected_labels = mnist_data['trainY'][selected_indices]


    class_type = set(mnist_data['trainY'])
    class_type = np.array(list(class_type), dtype=np)
    trainY = []
    trainX = []

    count = 0
    for i in class_type:
        trainY.append(np.where(selected_labels == i)[0])
        # testY.append(np.where(mnist_data['testY'] == i)[0])
        trainX.append(selected_samples[trainY[count], :])
        # testX.append(mnist_data['testX'][testY[count], :])
        count += 1


    data=trainX

    class_1 = trainX[0]
    class_2 = trainX[1]

    solution = train_all_classes(data, epsilon)

    testX = mnist_data['testX']
    testY = mnist_data['testY']

    predict, accuracy = test_classify_class(testX, testY, solution)
    print("Test Accuracy: ", accuracy)

    accuracies_random.append(accuracy)


plt.plot(label_percentages, accuracies_our_algo, label='Our Algo', marker='x', markersize=8)
plt.plot(label_percentages, accuracies_random, label='Random Selection', marker='o', markersize=8)
plt.ylim(0, 1)
plt.legend()
plt.xlabel("Label Percentage", fontsize=12)
plt.ylabel('Test Accuracy', fontsize=12)
plt.title("Label Selection on MNIST Data", fontsize=14)
plt.show()
