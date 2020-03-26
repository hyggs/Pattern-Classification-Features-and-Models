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


def initialize(digit):

    train_images_offset = 16
    train_labels_offset = 8
    n_digit = nump.zeros(10)
    for j in digit:
        for i in range(len(train_labels) - train_labels_offset):
            if train_labels[i + train_labels_offset] == digit[j]:
                n_digit[j] = n_digit[j] + 1

    return [n_digit, train_labels_offset, train_images_offset]


def extract(digit, d):

    [n_digit, train_labels_offset, train_images_offset] = initialize(digit)

    print("Extracting data...")
    x_pre = nump.zeros(0)
    for j in digit:
        r = 0
        c = 0
        x_pre = nump.zeros((10, int(n_digit[j]), d))
        for i in range(len(train_images) - train_images_offset):
            if train_labels[int((i - train_images_offset + train_labels_offset) / d)] == digit[j]:
                x_pre[j][r][c] = train_images[i + train_images_offset]
                c = c + 1
                if c == d:
                    c = 0
                    r = r + 1
                    if r == n_digit[j]:
                        print("end of array")

    print("Data extraction completes.")
    return [x_pre, n_digit]


def linear_regression():

    dimen = 784
    numbers = nump.arange(0, 10)
    xs, ns = extract(numbers, dimen)
    print(xs)
    print(ns)


linear_regression()
