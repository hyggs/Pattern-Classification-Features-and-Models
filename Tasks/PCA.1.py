import sys
import numpy as nump
import matplotlib.pyplot as plot
import gzip
with gzip.open('../MNIST/t10k-images-idx3-ubyte.gz', 'rb') as f:
    test_images = f.read()
with gzip.open('../MNIST/t10k-labels-idx1-ubyte.gz', 'rb') as f:
    test_labels = f.read()
with gzip.open('../MNIST/train-images-idx3-ubyte.gz', 'rb') as f:
    train_images = f.read()
with gzip.open('../MNIST/train-labels-idx1-ubyte.gz', 'rb') as f:
    train_labels = f.read()

nump.set_printoptions(threshold=sys.maxsize)


def initialize(digit):

    train_images_offset = 16
    train_labels_offset = 8
    n_digit = 0

    for i in range(len(train_labels) - train_labels_offset):
        if train_labels[i + train_labels_offset] == digit:
            n_digit = n_digit + 1

    return [n_digit, train_labels_offset, train_images_offset]


def extract(digit, d):

    [n_digit, train_labels_offset, train_images_offset] = initialize(digit)

    print("Extracting data...")

    x_pre = nump.zeros((n_digit, d))
    # print((len(train_labels) - 8) * 784) -> 47040000
    # print(len(train_images))             -> 47040016
    r4 = 0
    c4 = 0
    for i in range(len(train_images) - train_images_offset):
        if train_labels[int((i - train_images_offset + train_labels_offset) / d)] == 4:
            x_pre[r4][c4] = train_images[i + train_images_offset]
            c4 = c4 + 1
            if c4 == d:
                c4 = 0
                r4 = r4 + 1
                if r4 == n_digit:
                    print("end of array")

    print("Data extraction completes.")
    return [x_pre, n_digit]


# digit is the number in the current image, d is the feature dimension D, m is an array of projection dimension M.
def pca(digit, d, m):

    [x, n] = extract(digit, d)

    print("Initializing PCA parameters...")

    x_bar = nump.zeros(d)

    # Getting the mean vector x_bar.
    for i in range(d):
        for j in range(n):
            x_bar[i] += x[j][i]
        x_bar[i] = x_bar[i] / n

    # Getting the covariance matrix cov.
    # cov = nump.zeros((d, d))
    # for i in range(n):
    #     dev = nump.zeros(d)
    #     for j in range(d):
    #         dev[j] = x[i][j] - x_bar[j]
    #     cov += nump.outer(dev, dev)
    # cov = cov / n

    cov = nump.cov(x)

    # Eigenvalues and eigenvectors.
    lamb, u = nump.linalg.eig(cov)

    eigens = [(nump.abs(lamb[i]), u[:, i]) for i in range(len(lamb))]
    eigens.sort(key=lambda lam: lam[0], reverse=True)

    print("PCA parameters initialization completes.")

    # Calculating the distortion.
    distortion = nump.zeros(len(m))
    dimensionality = m
    for s in dimensionality:
        print("Working on dimension " + str(s) + "...")
        # projection_matrix = u[0:s]
        # sub_n = 0  # for n
        # for j in range(n):
        #     a = nump.subtract(x[j], x_bar)
        #     uu = 0
        #     for i in range(s):
        #         uu += nump.vdot(projection_matrix[i], projection_matrix[i])
        #     b = 1 - uu
        #     c = a * b
        #     d = nump.power(nump.linalg.norm(c), 2)
        #     sub_n += d
        # distortion[dimensionality.index(s)] = sub_n / n
        for i in range(s + 1, d):
            distortion[dimensionality.index(s)] += lamb[i]
        print("Dimension " + str(s) + " completes.")

    print("Distortion vector completes.")

    plot.plot(dimensionality, distortion, 'bo')
    plot.xlabel("M")
    plot.ylabel("J")
    plot.show()


ROW = 28
COLUMN = 28
D = ROW * COLUMN
M = [1, 2, 10, 50, 100, 200, 300, 400, 500, 600, 700, 784]
# M = [2, 10, 50, 100, 200, 300]
# M = [10]
pca(3, D, M)
# pca(4, D, M)
# pca(7, D)
# pca(8, D)
