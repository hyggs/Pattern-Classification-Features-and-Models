from mnist import MNIST
import numpy as nump
from sklearn.manifold import TSNE
import sys
import matplotlib.pyplot as plot

nump.set_printoptions(threshold=sys.maxsize)
nump.set_printoptions(precision=20)

print("Loading the data...")

data = MNIST("../data/MNIST/")

images, labels = data.load_training()
test_images, test_labels = data.load_testing()

zeros = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 0])
ones = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 1])
twos = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 2])
threes = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 3])
fours = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 4])
fives = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 5])
sixes = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 6])
sevens = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 7])
eights = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 8])
nines = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 9])
fours_sevens_eights = nump.array([images[key] for (key, label) in enumerate(labels)
                                  if int(label) == 4 or int(label) == 7 or int(label) == 8])
fours_sevens_eights_labels = nump.array([labels[key] for (key, label) in enumerate(labels)
                                         if int(label) == 4 or int(label) == 7 or int(label) == 8])

test_zeros = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if int(label) == 0])
test_ones = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if int(label) == 1])
test_twos = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if int(label) == 2])
test_threes = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if int(label) == 3])
test_fours = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if int(label) == 4])
test_fives = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if int(label) == 5])
test_sixes = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if int(label) == 6])
test_sevens = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if int(label) == 7])
test_eights = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if int(label) == 8])
test_nines = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if int(label) == 9])

all_data = nump.array([images[key] for (key, label) in enumerate(labels)])
all_label = nump.array([labels[key] for (key, label) in enumerate(labels)])


zeros_ones_training = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 0 or int(label) == 1])
zeros_ones_training_label = nump.array([labels[key] for (key, label) in enumerate(labels) if int(label) == 0 or
                                        int(label) == 1])

zeros_ones_testing = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if int(label) == 0 or
                                 int(label) == 1])
zeros_ones_testing_label = nump.array([test_labels[key] for (key, label) in enumerate(test_labels) if int(label) == 0 or
                                       int(label) == 1])

zeros_ones_twos_training = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 0 or
                                      int(label) == 1 or int(label) == 2])

zeros_ones_twos_training_labels = nump.array([labels[key] for (key, label) in enumerate(labels) if int(label) == 0 or
                                      int(label) == 1 or int(label) == 2])
zeros_ones_twos_testing = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if int(label) == 0 or
                                      int(label) == 1 or int(label) == 2])
zeros_ones_twos_testing_labels = nump.array([test_labels[key] for (key, label) in enumerate(test_labels) if int(label) == 0 or
                                      int(label) == 1 or int(label) == 2])


fours_sevens_eights_test_labels = nump.array([test_labels[key] for (key, label) in enumerate(test_labels) if
                                              int(label) == 4 or int(label) == 7 or int(label) == 8])
fours_sevens_eights_test = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if
                                      int(label) == 4 or int(label) == 7 or int(label) == 8])
zeros_twos_training = nump.array([images[key] for (key, label) in enumerate(labels) if
                                  int(label) == 0 or int(label) == 2])
zeros_twos_training_labels = nump.array([labels[key] for (key, label) in enumerate(labels) if
                                  int(label) == 0 or int(label) == 2])
zeros_twos_testing = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if
                                  int(label) == 0 or int(label) == 2])
ones_twos_training = nump.array([images[key] for (key, label) in enumerate(labels) if
                                        int(label) == 1 or int(label) == 2])

ones_twos_training_label = nump.array([labels[key] for (key, label) in enumerate(labels) if
                                              int(label) == 1 or int(label) == 2])

ones_twos_testing = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if
                                       int(label) == 1 or int(label) == 2])

ones_twos_testing_label = nump.array([test_labels[key] for (key, label) in enumerate(test_labels) if
                                             int(label) == 1 or int(label) == 2])
zeros_twos_testing_labels = nump.array([test_labels[key] for (key, label) in enumerate(test_labels) if
                                  int(label) == 0 or int(label) == 2])
ones_twos_threes_training = nump.array([images[key] for (key, label) in enumerate(labels) if
                                        int(label) == 1 or int(label) == 2 or int(label) == 3])

ones_twos_threes_training_label = nump.array([labels[key] for (key, label) in enumerate(labels) if
                                              int(label) == 1 or int(label) == 2 or int(label) == 3])

ones_twos_threes_testing = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if
                                       int(label) == 1 or int(label) == 2 or int(label) == 3])

ones_twos_threes_testing_label = nump.array([test_labels[key] for (key, label) in enumerate(test_labels) if
                                             int(label) == 1 or int(label) == 2 or int(label) == 3])
train_0123456789 = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 0 or
                        int(label) == 1 or int(label) == 2 or int(label) == 3 or int(label) == 4 or
                             int(label) == 5 or int(label) == 6 or int(label) == 7 or int(label) == 8])
train_0123456789_label = nump.array([labels[key] for (key, label) in enumerate(labels) if int(label) == 0 or
                               int(label) == 1 or int(label) == 2 or int(label) == 3 or int(label) == 4 or
                             int(label) == 5 or int(label) == 6 or int(label) == 7 or int(label) == 8])
test_0123456789 = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if int(label) == 0 or
                        int(label) == 1 or int(label) == 2 or int(label) == 3 or int(label) == 4 or
                             int(label) == 5 or int(label) == 6 or int(label) == 7 or int(label) == 8])
test_0123456789_label = nump.array([test_labels[key] for (key, label) in enumerate(test_labels) if int(label) == 0 or
                              int(label) == 1 or int(label) == 2 or int(label) == 3 or int(label) == 4 or
                             int(label) == 5 or int(label) == 6 or int(label) == 7 or int(label) == 8])

print("Data loading completes.")

print("FC Neural Networks starts...")

M = 784
X = nump.array(images)
N = int(len(X))
X = X[0:N, 0:M]
X = X / 2550
Y = nump.array(labels)
TX = nump.array(test_images)
TX = TX[:, 0:M]
TX = TX / 2550
TY = nump.array(test_labels)
L1 = 5.5
L2 = 4.0
N1 = 15
C = 10
Weights1 = nump.random.rand(M, N1)  # Weights1 is M by N1.
Weights2 = nump.random.rand(N1, C)  # Weights2 is N1 by C.
Layer0 = nump.ones((M, 1))  # Layer0 is M by 1.
Layer1 = nump.ones((N1, 1))  # Layer1 is N1 by 1.
Layer2 = nump.ones((C, 1))  # Layer2 is C by 1.
In1 = nump.random.rand(N1, 1)   # Inputs for Layer1 is N1 by 1.
Out1 = nump.random.rand(N1, 1)  # Outputs from Layer1 is N1 by 1.
In2 = nump.random.rand(C, 1)    # Inputs for Layer2 is C by 1.
Out2 = nump.random.rand(C, 1)   # Outputs from Layer2 is C by 1.
EPOCH = 9
for e in range(EPOCH):
    print("FC Neural Networks running on EPOCH " + str(e) + "...")
    for i in range(len(X)):
        Error = nump.zeros((C, 1))
        derivatives1 = nump.zeros((M, N1))
        derivatives2 = nump.zeros((N1, C))
        # forward
        Layer0 = X[i, :].reshape(M, 1)
        In1 = nump.dot(Layer0.T, Weights1).T
        Out1 = 1 / (1 + nump.exp(-In1))
        In2 = nump.dot(Out1.T, Weights2).T
        Out2 = 1 / (1 + nump.exp(-In2))

        # backward
        if Y[i] == 0:
            Error[0] = Out2[0] - 1
            Error[1] = Out2[1]
            Error[2] = Out2[2]
            Error[3] = Out2[3]
            Error[4] = Out2[4]
            Error[5] = Out2[5]
            Error[6] = Out2[6]
            Error[7] = Out2[7]
            Error[8] = Out2[8]
            Error[9] = Out2[9]
        if Y[i] == 1:
            Error[0] = Out2[0]
            Error[1] = Out2[1] - 1
            Error[2] = Out2[2]
            Error[3] = Out2[3]
            Error[4] = Out2[4]
            Error[5] = Out2[5]
            Error[6] = Out2[6]
            Error[7] = Out2[7]
            Error[8] = Out2[8]
            Error[9] = Out2[9]
        if Y[i] == 2:
            Error[0] = Out2[0]
            Error[1] = Out2[1]
            Error[2] = Out2[2] - 1
            Error[3] = Out2[3]
            Error[4] = Out2[4]
            Error[5] = Out2[5]
            Error[6] = Out2[6]
            Error[7] = Out2[7]
            Error[8] = Out2[8]
            Error[9] = Out2[9]
        if Y[i] == 3:
            Error[0] = Out2[0]
            Error[1] = Out2[1]
            Error[2] = Out2[2]
            Error[3] = Out2[3] - 1
            Error[4] = Out2[4]
            Error[5] = Out2[5]
            Error[6] = Out2[6]
            Error[7] = Out2[7]
            Error[8] = Out2[8]
            Error[9] = Out2[9]
        if Y[i] == 4:
            Error[0] = Out2[0]
            Error[1] = Out2[1]
            Error[2] = Out2[2]
            Error[3] = Out2[3]
            Error[4] = Out2[4] - 1
            Error[5] = Out2[5]
            Error[6] = Out2[6]
            Error[7] = Out2[7]
            Error[8] = Out2[8]
            Error[9] = Out2[9]
        if Y[i] == 5:
            Error[0] = Out2[0]
            Error[1] = Out2[1]
            Error[2] = Out2[2]
            Error[3] = Out2[3]
            Error[4] = Out2[4]
            Error[5] = Out2[5] - 1
            Error[6] = Out2[6]
            Error[7] = Out2[7]
            Error[8] = Out2[8]
            Error[9] = Out2[9]
        if Y[i] == 6:
            Error[0] = Out2[0]
            Error[1] = Out2[1]
            Error[2] = Out2[2]
            Error[3] = Out2[3]
            Error[4] = Out2[4]
            Error[5] = Out2[5]
            Error[6] = Out2[6] - 1
            Error[7] = Out2[7]
            Error[8] = Out2[8]
            Error[9] = Out2[9]
        if Y[i] == 7:
            Error[0] = Out2[0]
            Error[1] = Out2[1]
            Error[2] = Out2[2]
            Error[3] = Out2[3]
            Error[4] = Out2[4]
            Error[5] = Out2[5]
            Error[6] = Out2[6]
            Error[7] = Out2[7] - 1
            Error[8] = Out2[8]
            Error[9] = Out2[9]
        if Y[i] == 8:
            Error[0] = Out2[0]
            Error[1] = Out2[1]
            Error[2] = Out2[2]
            Error[3] = Out2[3]
            Error[4] = Out2[4]
            Error[5] = Out2[5]
            Error[6] = Out2[6]
            Error[7] = Out2[7]
            Error[8] = Out2[8] - 1
            Error[9] = Out2[9]
        if Y[i] == 9:
            Error[0] = Out2[0]
            Error[1] = Out2[1]
            Error[2] = Out2[2]
            Error[3] = Out2[3]
            Error[4] = Out2[4]
            Error[5] = Out2[5]
            Error[6] = Out2[6]
            Error[7] = Out2[7]
            Error[8] = Out2[8]
            Error[9] = Out2[9] - 1
        pd0 = Error[0] * Out2[0] * (1 - Out2[0])
        pd1 = Error[1] * Out2[1] * (1 - Out2[1])
        pd2 = Error[2] * Out2[2] * (1 - Out2[2])
        pd3 = Error[3] * Out2[3] * (1 - Out2[3])
        pd4 = Error[4] * Out2[4] * (1 - Out2[4])
        pd5 = Error[5] * Out2[5] * (1 - Out2[5])
        pd6 = Error[6] * Out2[6] * (1 - Out2[6])
        pd7 = Error[7] * Out2[7] * (1 - Out2[7])
        pd8 = Error[8] * Out2[8] * (1 - Out2[8])
        pd9 = Error[9] * Out2[9] * (1 - Out2[9])
        for d in range(len(derivatives2)):
            derivatives2[d, 0] = pd0 * Out1[d]
            derivatives2[d, 1] = pd1 * Out1[d]
            derivatives2[d, 2] = pd2 * Out1[d]
            derivatives2[d, 3] = pd3 * Out1[d]
            derivatives2[d, 4] = pd4 * Out1[d]
            derivatives2[d, 5] = pd5 * Out1[d]
            derivatives2[d, 6] = pd6 * Out1[d]
            derivatives2[d, 7] = pd7 * Out1[d]
            derivatives2[d, 8] = pd8 * Out1[d]
            derivatives2[d, 9] = pd9 * Out1[d]
        pdf = nump.zeros(N1)
        for f in range(N1):
            pdf[f] = (pd0 * Weights2[f][0] + pd1 * Weights2[f][1] + pd2 * Weights2[f][2] + pd3 * Weights2[f][3] +
                      pd4 * Weights2[f][4] + pd5 * Weights2[f][5] + pd6 * Weights2[f][6] + pd7 * Weights2[f][7] +
                      pd8 * Weights2[f][8] + pd9 * Weights2[f][9]) * Out1[f] * (1 - Out1[f])
        for m in range(M):
            derivatives1[m, :] = pdf * X[i][m]

        Weights2 = Weights2 - L2 * derivatives2
        Weights1 = Weights1 - L1 * derivatives1

    print("FC Neural Networks testing...")

    S = 0
    F = 0
    for i in range(len(TX)):
        Layer0 = TX[i, :].reshape(M, 1)
        In1 = nump.dot(Layer0.T, Weights1).T
        Out1 = 1 / (1 + nump.exp(-In1))
        In2 = nump.dot(Out1.T, Weights2).T
        Out2 = 1 / (1 + nump.exp(-In2))
        if TY[i] == 0 and Out2[0] == nump.max(Out2):
            S += 1
        elif TY[i] == 1 and Out2[1] == nump.max(Out2):
            S += 1
        elif TY[i] == 2 and Out2[2] == nump.max(Out2):
            S += 1
        elif TY[i] == 3 and Out2[3] == nump.max(Out2):
            S += 1
        elif TY[i] == 4 and Out2[4] == nump.max(Out2):
            S += 1
        elif TY[i] == 5 and Out2[5] == nump.max(Out2):
            S += 1
        elif TY[i] == 6 and Out2[6] == nump.max(Out2):
            S += 1
        elif TY[i] == 7 and Out2[7] == nump.max(Out2):
            S += 1
        elif TY[i] == 8 and Out2[8] == nump.max(Out2):
            S += 1
        elif TY[i] == 9 and Out2[9] == nump.max(Out2):
            S += 1
        else:
            F += 1
    print("EPOCH " + str(e))
    print("Success: " + str(S))
    print("Failure: " + str(F))
    print("Success Rate: " + str(S / (S + F)))
    L1 += 1
    L2 += 1
print("FC Neural Networks ends.")
