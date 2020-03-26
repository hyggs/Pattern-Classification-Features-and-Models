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
        # print(int(i - train_images_offset + train_labels_offset / d))
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
    # print(len(x))
    # print(len(x[0]))

    # Getting the mean vector x_bar.
    for i in range(d):
        for j in range(n):
            x_bar[i] += x[j][i]
        x_bar[i] = x_bar[i] / n

    # Getting the covariance matrix cov.
    cov = nump.zeros((d, d))
    for i in range(n):
        dev = nump.zeros(d)
        for j in range(d):
            dev[j] = x[i][j] - x_bar[j]
        cov += nump.outer(dev, dev)
    cov = cov / n

    # Eigenvalues and eigenvectors.
    [lamb, u] = nump.linalg.eig(cov)
    sorted_lamb_indices = nump.argsort(lamb)
    # print("sorted lamb indices")
    # print(sorted_lamb_indices)
    sorted_lamb_indices_reversed = nump.flip(sorted_lamb_indices)
    # print("sorted_lamb_indices_reversed")
    # print(sorted_lamb_indices_reversed)
    largest_m_lamb = nump.zeros(d, dtype=complex)
    largest_m_u = nump.zeros((n, d), dtype=complex)

    print("PCA parameters initialization completes.")

    # Calculating the distortion.
    distortion = nump.zeros(len(m))
    dimensionality = m
    for s in dimensionality:
        print("Working on dimension " + str(s) + "...")
        # x_u = nump.zeros(d)
        # x_bar_u = nump.zeros(d)
        # large_x_bar = nump.repeat(x_bar, n)
        largest_m_indices = sorted_lamb_indices_reversed[0:s]
        largest_m_lamb[0:s] = lamb[largest_m_indices]

        print(largest_m_lamb[0:s])
        largest_m_u[0:s] = u[largest_m_indices]
        print(largest_m_u[0:s])

        sub_n = 0  # for n
        for j in range(n):
            sub_s = 0   # for s
            for i in range(s):
                a = nump.vdot(x[j], largest_m_u[i])
                b = nump.vdot(x_bar, largest_m_u[i])
                c = nump.subtract(a, b)
                d = c * largest_m_u[i]
                sub_s += d
            e = nump.add(x_bar, sub_s)
            g = nump.subtract(x[j], e)
            h = nump.square(nump.absolute(g))
            sub_n += nump.sum(h)
        distortion[dimensionality.index(s)] = sub_n / n
        print("Dimension " + str(s) + " completes.")

    # print(u[0:m])
    # print(largest_m_u[0:m])

    # Sum of the dot product of each eigenvector with itself, up to i-dimension, stored in an array sum_uu.
    # sum_uu = nump.zeros(d, dtype=complex)
    # sum_uu[0] = nump.vdot(largest_m_u[0], largest_m_u[0])
    # for i in range(1, m):
    #     sum_uu[i] = nump.add(sum_uu[i - 1], nump.vdot(largest_m_u[i], largest_m_u[i]))
    #
    # print("Calculating distortions...")
    # # Filling in the array of distortion.
    # for s in range(m):
    #     print("Working on dimension " + str(m) + "...")
    #     for j in range(n):
    #         distortion[s] += nump.sum(nump.square(nump.absolute(nump.subtract(x[j], x_bar) * (1 - sum_uu[s]))))
    #     distortion[s] = distortion[s] / n
    #     print("Dimension " + str(m) + " completes.")
    # print(sum_uu)

# -----------------------------
#     for s in range(m):
#         for i in range(m, d):
#             distortion[s] += largest_m_lamb[i]
#     for s in [0, 1, 9, 49, 99, 199, 299, 783]:
#         print("Working on dimension " + str(s) + "...")
#         sub_n = 0  # for n
#         for j in range(n):
#             sub_s = 0   # for s
#             for i in range(s):
#                 a = nump.vdot(x[j], largest_m_u[i])
#                 b = nump.vdot(x_bar, largest_m_u[i])
#                 c = nump.subtract(a, b)
#                 d = c * largest_m_u[i]
#                 sub_s += d
#             e = nump.add(x_bar, sub_s)
#             g = nump.subtract(x[j], e)
#             h = nump.square(nump.absolute(g))
#             sub_n += nump.sum(h)
#         distortion[s] = sub_n / n
#         print("Dimension " + str(s) + " completes.")

    # ----------------------------------
    print("Distortion vector completes.")

    # print(sub_n)
    # x_uu = nump.vdot(nump.vdot(x, largest_i_u), largest_i_u)
    # x_bar_uu = nump.vdot(nump.vdot(large_x_bar, largest_i_u), largest_i_u)
    # subtracted = nump.subtract(x_uu, x_bar_uu)

    # error = 0
    # x_projected = nump.zeros(d)
    # for j in range(n):
    #     subsum = nump.zeros(d)

    #     for t in range(i):
    #         x_t_u_i = nump.vdot(x[j], u[largest_m_indices[t]])
    #         subsum = nump.add(subsum, nump.dot(nump.subtract(x_t_u_i, x_bar_t_u_i), u[largest_m_indices[t]]))
    #     x_projected[j] = nump.add(x_bar, subsum)
    #     error += nump.square(nump.linalg.norm(x[j] - x_projected[j]))
    # distortion[i] = error / n

    plot.plot(dimensionality, distortion, 'bo')
    plot.xlabel("M")
    plot.ylabel("J")
    plot.show()


ROW = 28
COLUMN = 28
D = ROW * COLUMN
# M = [1, 2, 10, 50, 100, 200, 300, 400, 500, 600, 700, 784]
# M = [2, 10, 50, 100, 200, 300]
M = [10]
pca(3, D, M)
# pca(4, D, M)
# pca(7, D)
# pca(8, D)

