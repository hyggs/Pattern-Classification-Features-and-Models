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
X = nump.array(images)
X = X[0:15000, :]
Y = nump.array(labels)
Y = Y[0:15000]
TX = nump.array(test_images)
TY = nump.array(test_labels)
M = 784
L = 0.6     # Learning rate.
IT = 100
print("Kernel SVM starts...")


def kernel(x):
    r = 0
    for k in range(len(X)):
        r += nump.exp(-gamma * pow(nump.linalg.norm(x - X[j]), 2))
    return r


gamma = nump.random.rand()
K = nump.zeros((len(X), len(X)))
for i in range(len(X)):
    for j in range(len(X)):
        nom = nump.linalg.norm(X[i] - X[j])
        K[i][j] = nump.exp(-gamma * pow(nom, 2))
Q = nump.dot(nump.dot(Y, Y.T), K)
alpha = nump.random.rand(len(X), 1)

print("Training kernel...")

e = nump.repeat(1, len(X))
for i in range(IT):
    g = nump.dot(Q, alpha) - e
    alpha = alpha - g * L

print("Testing...")

for i in range(len(TX)):
    P = kernel(TX[i]) * TY[i] * alpha
    print(P)

print("Linear Soft-SVM completes.")
