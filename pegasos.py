import os
import csv
import math
import time
import pickle
import numpy as np
from sklearn import svm
from numpy import linalg as LA
from sklearn.metrics import accuracy_score


def read(filename):
    features = list()
    classlabels = list()
    with open(filename) as fh:		
        reader = csv.reader(fh)
        for datanum, data in enumerate(reader):
            if datanum > 2000:
                break
            if len(data) and datanum:
                vector = [float(datapoint) for index, datapoint in enumerate(data[1:])]
                classlabel = int(data[0]) #1 if int(data[0]) == 0 else -1
                features.append(vector)
                classlabels.append(classlabel)
    return np.array(features), np.array(classlabels)

def computeGRAM(X_train, Kernel):
    gram = np.zeros((X_train.shape[0], X_train.shape[0]))
    for i in range(X_train.shape[0]):
        for j in range(i, X_train.shape[0]):		
            gram[i,j] = Kernel(X_train[i], X_train[j])
    for i in range(X_train.shape[0]):
        for j in range(i):
            gram[i,j] = gram[j,i]
    return gram

def main(TrainingFilename, TestingFilename, Iterations = 5, Kernel = "", lambdaa = .001):
    X_train, Y_train=read(TrainingFilename)
    print("Loaded Training Samples")

    X_test, Y_test=read(TestingFilename)
    print("Loaded Testing Samples")

    if Kernel:
        print("Computing Gram Matrix")
        GramMatrix = computeGRAM(X_train, Kernel)
        print("Computed Gram Matrix")
    else:
        GramMatrix = np.array([])

    print("Computing %d Iterations of Pegasos" % Iterations)
    start = time.time()
    if np.unique(Y_train).shape[0] > 2:
        Coeffecients_list = list()
        Supportx_list = list()
        for i in np.unique(Y_train):
            Y_train_modified = np.copy(Y_train)
            Y_train_modified[Y_train_modified==i] = 1
            Y_train_modified[Y_train_modified!=i] = -1
            Coeffecients, Supportx = Pegasos(X_train, Y_train_modified, GramMatrix, lambdaa, Iterations)
            Coeffecients_list.append(Coeffecients)
            Supportx_list.append(Supportx)
    else:
        Coeffecients, Supportx = Pegasos(X_train, Y_train, GramMatrix, lambdaa, Iterations)
    end = time.time()
    print('Completed Pegasos')
    print("The training time of Pegasos with %d Iterations is %f" % (Iterations, end-start))

    start = time.time()
    if np.unique(Y_train).shape[0] > 2:
        Y_predict = np.zeros(Y_test.shape[0])
        for num, i in enumerate(np.unique(Y_train)):
            Y_predict_i = RunTests(Coeffecients_list[num], Supportx_list[num], Kernel, X_test)
            Y_predict[Y_predict_i==1] = i
    else:
        Y_predict = RunTests(Coeffecients, Supportx, Kernel, X_test)
    end = time.time()
    print("The time required to predict %d of samples is %f" % (Y_test.shape[0], end - start))
    print("The accuracy of implemented Pegasos is %f" % accuracy_score(Y_test, Y_predict))
    '''
    print("Running sklearn SVM")
    clf = svm.SVC(C = 0.001, gamma='scale', kernel = 'poly')
    clf.fit(X_train, Y_train)
    Y_predict = clf.predict(X_test)
    print("The accuracy of sklearn SVM is %f" % accuracy_score(Y_test, Y_predict))
    '''

def Pegasos(X_train, Y_train, GramMatrix, lambdaa = 0.1, Iterations = 5):
    samples = X_train.shape[0]
    if GramMatrix.shape[0] > 0:
        w = np.zeros(samples)
    else:
        w = np.random.rand(X_train.shape[1])
    time = 1
    for i in range(Iterations):
        for tau in range(len(X_train)):
            if GramMatrix.shape[0] == 0:
                etat = 1/(time*lambdaa)  
                wx = np.dot(w, X_train[tau])
                w *= (1-1/time)
                if Y_train[tau] * wx < 1:
                    w += etat * Y_train[tau] * X_train[tau]
                # w *= min(1, 1/(math.sqrt(lambdaa)*LA.norm(w)))
            else:
                wx = np.dot(w, GramMatrix[tau])
                w *= (1-1/time)
                if(Y_train[tau]*wx < 1):
                    w[tau] += Y_train[tau]  / (1/lambdaa)
            time += 1
    non_zero_indices = np.where(w != 0)
    SV = X_train[non_zero_indices, :]
    if GramMatrix.shape[0] > 0:
        w = w[non_zero_indices]
    return np.squeeze(w), np.squeeze(SV)

def Predictor(w, Sample, SV, Kernel=""):
    if Kernel:
        accumulator = 0	
        for index in range(w.shape[0]):
            accumulator += w[index]*Kernel(SV[index], Sample)
    else:
        accumulator = np.dot(w, Sample)
    if accumulator < 0: return -1
    return 1

def RunTests(w, SV, Kernel, TestingSamples):
    Y_predict = list()
    for Sample in TestingSamples:
        predictedlabel=Predictor(w, Sample, SV, Kernel)
        Y_predict.append(predictedlabel)
    return np.asarray(Y_predict)
    
def radial_basis(gamma):
    def function(vector1, vector2):
        new_vector = vector1 - vector2
        return math.exp(-1 * gamma * LA.norm(new_vector) ** 2)
    return function

def homogeneous_polynomial(degree):
    def function(vector1, vector2):
        value = np.dot(vector1, vector2)		
        return value ** degree
    return function

def inhomogeneous_polynomial(degree):
    def function(vector1, vector2):
        value = np.dot(vector1, vector2) + 1
        return value ** degree
    return function

def linear(): return homogeneous_polynomial(1)

def hyperbolic_tangent(kappa, c):
    def function(vector1, vector2):
        value = np.dot(vector1, vector2)	
        return math.tanh(kappa*value+c)
    return function

if __name__ == "__main__":
    trainingfile = "fashion-mnist_train_binary.csv"
    testingfile = "fashion-mnist_test_binary.csv"
    main(trainingfile, testingfile, 5, linear())