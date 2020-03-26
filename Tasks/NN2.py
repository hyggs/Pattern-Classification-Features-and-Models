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


print("Data loading completes.")

print("FC Neural Networks starts...")

M = 784
X = zeros_ones_training
N = int(len(X))
X = X[0:N, 0:M]
X = X / 2550
Y = zeros_ones_training_label
TX = zeros_ones_testing
TX = TX[:, 0:M]
TX = TX / 2550
TY = zeros_ones_testing_label
L1 = 2.5
L2 = 1.5
N1 = 12
C = 2
Weights1 = nump.random.rand(M, N1)  # Weights1 is M by N1.
Weights2 = nump.random.rand(N1, C)  # Weights2 is N1 by 2.
Layer0 = nump.ones((M, 1))  # Layer0 is M by 1.
Layer1 = nump.ones((N1, 1))  # Layer1 is N1 by 1.
Layer2 = nump.ones((C, 1))  # Layer2 is 2 by 1.
In1 = nump.random.rand(N1, 1)   # Inputs for Layer1 is N1 by 1.
Out1 = nump.random.rand(N1, 1)  # Outputs from Layer1 is N1 by 1.
In2 = nump.random.rand(C, 1)    # Inputs for Layer2 is 2 by 1.
Out2 = nump.random.rand(C, 1)   # Outputs from Layer2 is 2 by 1.
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
    if Y[i] == 1:
        Error[0] = Out2[0]
        Error[1] = Out2[1] - 1
    for d in range(len(derivatives2)):
        derivatives2[d, 0] = Error[0] * Out2[0] * (1 - Out2[0]) * Out1[d]
        derivatives2[d, 1] = Error[1] * Out2[1] * (1 - Out2[1]) * Out1[d]
    for d in range(M):
        for f in range(N1):
            derivatives1[d, f] = (Error[0] * Out2[0] * (1 - Out2[0]) * Weights2[f][0] +
                                  Error[1] * Out2[1] * (1 - Out2[1]) * Weights2[f][1]) \
                                 * Out1[f] * (1 - Out1[f]) * X[i][d]
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
    for n in range(C):
        if TY[i] == n and Out2[n] == nump.max(Out2):
            S += 1
        else:
            F += 1

print(S)
print(F)
print("FC Neural Networks ends.")