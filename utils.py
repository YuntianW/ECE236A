import csv
import os
import numpy as np
import matplotlib.pyplot as plt

def prepare_synthetic_data():
    data = dict()

    X = np.loadtxt('Data/synthetic/synthetic_X.csv', delimiter=',').reshape(1500, 2)
    Y = np.loadtxt('Data/synthetic/synthetic_Y.csv', delimiter=',')

    data['trainX'] = X[:1000]  # 1000 x 2
    data['trainY'] = Y[:1000]  # 1000 x 1
    data['testX']  = X[1000:]  # 500 x 2
    data['testY']  = Y[1000:]  # 500 x 1
 
    return data


def prepare_mnist_data():
    data = dict()

    X = np.loadtxt('Data/reduced_mnist/mnist_X.csv', delimiter=',').reshape(1500, 784)
    Y = np.loadtxt('Data/reduced_mnist/mnist_Y.csv', delimiter=',')

    data['trainX'] = X[:1000]  # 1000 x 784
    data['trainY'] = Y[:1000]  # 1000 x 1
    data['testX']  = X[1000:]  # 500 x 784
    data['testY']  = Y[1000:]  # 500 x 1

    return data


def plot_result(result1 = None, result2 = None, result3 = None):
    ''' Input Format with Examples:
        from Task1: result1 = {'synthetic_test_accuracy':0.9, 'mnist_test_accuracy':0.85}

        from Task2: result2 = {'synthetic':{'K':[3, 5, 10], 'clustering_nmi':[0.6,0.6,0.6], 'classification_accuracy':[0.8,0.8,0.8]},
            'mnist':{'K':[3, 10, 32], 'clustering_nmi':[0.5,0.5,0.5], 'classification_accuracy':[0.7,0.7,0.7]}}
                                
        from Task3: result3 = {'synthetic': {'label_percentage':[0.05,0.1,0.2,0.5,1], 'test_accuracy(our algo)':[0.5,0.6,0.7,0.8,0.9], 'test_accuracy(random)':[0.4,0.5,0.6,0.7,0.8]},
            'mnist': {'label_percentage':[0.05,0.1,0.2,0.5,1], 'test_accuracy(our algo)':[0.5,0.5,0.7,0.7,0.7], 'test_accuracy(random)':[0.4,0.4,0.6,0.6,0.6]}}
    '''

    if result1 != None and result2 != None:
        x1 = [-0.4,0,0.4]
        x2 = [1,1.4,1.8]
        width = 0.35

        plt.bar(x1, result2['synthetic']['clustering_nmi'], width)
        for i in range(3):
            plt.text(x1[i], result2['synthetic']['clustering_nmi'][i]/2, 'K={}'.format(result2['synthetic']['K'][i]), fontsize=12, ha='center', bbox = dict(facecolor = 'white', alpha = .8))
        plt.bar(x2, result2['mnist']['clustering_nmi'], width)
        for i in range(3):
            plt.text(x2[i], result2['mnist']['clustering_nmi'][i]/2, 'K={}'.format(result2['mnist']['K'][i]), fontsize=12, ha='center', bbox = dict(facecolor = 'white', alpha = .8))
        plt.ylim(0,1)
        plt.xticks([0,1.3], ['Synthetic', 'MNIST'], fontsize=12)
        plt.ylabel('NMI', fontsize=12)
        plt.xlabel('Dataset', fontsize=12)
        plt.title("Clustering Performance on training data", fontsize=14)
        plt.show()


        plt.bar(x1, result2['synthetic']['classification_accuracy'], width)
        for i in range(3):
            plt.text(x1[i], result2['synthetic']['classification_accuracy'][i]/2, 'K={}'.format(result2['synthetic']['K'][i]), fontsize=12, ha='center', bbox = dict(facecolor = 'white', alpha = .8))
        plt.bar(x2, result2['mnist']['classification_accuracy'], width)
        for i in range(3):
            plt.text(x2[i], result2['mnist']['classification_accuracy'][i]/2, 'K={}'.format(result2['mnist']['K'][i]), fontsize=12, ha='center', bbox = dict(facecolor = 'white', alpha = .8))

        plt.ylim(0,1)
        plt.xticks([0,1.3], ['Synthetic', 'MNIST'], fontsize=12)
        plt.axhline(y=result1['synthetic_test_accuracy'], color='blue', linestyle='--', linewidth=2, label = 'supervised (Synthetic)')
        plt.axhline(y=result1['mnist_test_accuracy'], color='orange', linestyle='--', linewidth=2, label = 'supervised (MNIST)')
        plt.legend(bbox_to_anchor=(1, 1))
        plt.title("Classification Performance on testing data", fontsize=14)
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.xlabel('Dataset', fontsize=12)
        plt.show()
    

    if result3 != None: 
        plt.plot(result3['synthetic']['label_percentage'], result3['synthetic']['test_accuracy(our algo)'], label='our algo', marker='x', markersize=8)
        plt.plot(result3['synthetic']['label_percentage'], result3['synthetic']['test_accuracy(random)'], label='random selection', marker='o', markersize=8)
        plt.ylim(0,1)
        plt.legend()
        plt.xlabel("Label Percentage", fontsize=12) 
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.title("Label Selection on Synthetic Data", fontsize=14)
        plt.show()

        plt.plot(result3['mnist']['label_percentage'], result3['mnist']['test_accuracy(our algo)'], label='our algo', marker='x', markersize=8,linewidth=2)
        plt.plot(result3['mnist']['label_percentage'], result3['mnist']['test_accuracy(random)'], label='random selection', marker='o', markersize=8, linewidth=2)
        plt.ylim(0,1)
        plt.legend()
        plt.xlabel("Label Percentage", fontsize=12) 
        plt.ylabel('Test Accuracy', fontsize=12)
        plt.title("Label Selection on MNIST Data", fontsize=14)
        plt.show()