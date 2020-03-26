import sys
import numpy as nump
import matplotlib.pyplot as plot
import matplotlib.cm as cm
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

train_images_offset = 16
train_labels_offset = 8
n_digit = nump.zeros(10)
digit = nump.arange(0, 10)

for j in digit:
    for i in range(len(train_labels) - train_labels_offset):
        if train_labels[i + train_labels_offset] == digit[j]:
            n_digit[j] = n_digit[j] + 1

ROW = 28
COLUMN = 28
D = ROW * COLUMN
M = [2, 10, 50, 100, 200, 300]

X4 = nump.zeros((n_digit[4], D))
ROW4 = 0
COL4 = 0
for i in range(len(train_images) - train_images_offset):
    if train_labels[int((i - train_images_offset + train_labels_offset) / D)] == 4:
        X4[ROW4][COL4] = train_images[i + train_images_offset]
        COL4 = COL4 + 1
        if COL4 == 4:
            OCL4 = 0
            ROW4 = ROW4 + 1


# digit is the number in the current image, d is the feature dimension D, m is an array of projection dimension M.
def pca(number, d, m):

    print("Initializing PCA parameters...")

    x_bar = nump.zeros(d)

    # Getting the mean vector x_bar.
    for i in range(d):
        for j in range(n_digit[number]):
            x_bar[i] += [j][i]
        x_bar[i] = x_bar[i] / n

    cov = nump.cov(x4)

    # Eigenvalues and eigenvectors.
    lamb, u = nump.linalg.eig(cov)

    eigens = [(nump.abs(lamb[i]), u[:, i]) for i in range(len(lamb))]
    eigens.sort(key=lambda lam: lam[0], reverse=True)

    print("PCA parameters initialization completes.")

    # Calculating the distortion.
    distortion = nump.zeros(len(m))
    dimensionality = m
    for s in dimensionality:
        for i in range(s + 1, d):
            distortion[dimensionality.index(s)] += nump.abs(lamb[i])

    print("Distortion vector completes.")

    plot.plot(dimensionality, distortion, 'bo')
    plot.xlabel("M")
    plot.ylabel("J")
    plot.show()

    # Projecting x.
    # chosen_m = 300
    # projection_matrix = u[0:chosen_m]
    # projection = nump.dot(nump.abs(projection_matrix), x)
    # print(projection)
    # plot.imsave('projected_digit_' + str(digit), nump.array(projection, 3))
    # plot.imshow(nump.array(projection, 3))



# print("Running PCA on training images of digit 4...")
# pca(4, D, M)
# print("PCA on training images of digit 4 completes.")
# print("Running PCA on training images of digit 7...")
# pca(7, D, M)
# print("PCA on training images of digit 7 completes.")
# print("Running PCA on training images of digit 8...")
# pca(8, D, M)
# print("PCA on training images of digit 8 completes.")
