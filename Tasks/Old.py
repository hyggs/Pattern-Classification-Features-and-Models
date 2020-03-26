from mnist import MNIST
import numpy as nump
import tsne_python.tsne_python.tsne as tSNE
import sys
import matplotlib.pyplot as plot

nump.set_printoptions(threshold=sys.maxsize)

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

print("Data loading completes.")

print("PCA starts...")

# Each matrix of number is n by d where n = number of numbers and d is the dimension = 28 * 28 = 784.

# for i in range(len(sevens)):
#     x4 = nump.array(sevens[i]).reshape(28, 28)
#     plot.imshow(x4, cmap='gray')
#     plot.show()

mean_zero = nump.mean(zeros, axis=0)
mean_one = nump.mean(ones, axis=0)
mean_four = nump.mean(fours, axis=0)
mean_seven = nump.mean(sevens, axis=0)
mean_eight = nump.mean(eights, axis=0)
mean_four_seven_eight = nump.mean(fours_sevens_eights, axis=0)


cov_zero = nump.cov(zeros.T)
cov_one = nump.cov(ones.T)
cov_four = nump.cov(fours.T)
cov_seven = nump.cov(sevens.T)
cov_eight = nump.cov(eights.T)
cov_four_seven_eight = nump.cov(fours_sevens_eights.T)

eigenvalues_four, eigenvectors_four = nump.linalg.eig(cov_four)
eigenvalues_seven, eigenvectors_seven = nump.linalg.eig(cov_seven)
eigenvalues_eight, eigenvectors_eight = nump.linalg.eig(cov_eight)
eigenvalues_four_seven_eight, eigenvectors_four_seven_eight = nump.linalg.eig(cov_four_seven_eight)

sorted_four = eigenvalues_four.argsort()[::-1]
sorted_seven = eigenvalues_seven.argsort()[::-1]
sorted_eight = eigenvalues_eight.argsort()[::-1]
sorted_four_seven_eight = eigenvalues_four_seven_eight.argsort()[::-1]

eigenvalues_four = eigenvalues_four[sorted_four]
eigenvalues_seven = eigenvalues_seven[sorted_seven]
eigenvalues_eight = eigenvalues_eight[sorted_eight]
eigenvalues_four_seven_eight = eigenvalues_four_seven_eight[sorted_four_seven_eight]

eigenvectors_four = eigenvectors_four[sorted_four]
eigenvectors_seven = eigenvectors_seven[sorted_seven]
eigenvectors_eight = eigenvectors_eight[sorted_eight]
eigenvectors_four_seven_eight = eigenvectors_four_seven_eight[sorted_four_seven_eight]

# Calculating the distortion for each dimension.
D = 784
dimensionality = [2, 10, 50, 100, 200, 300]
# distortion_four = nump.zeros(len(dimensionality))
# distortion_seven = nump.zeros(len(dimensionality))
# distortion_eight = nump.zeros(len(dimensionality))
# distortion_four_seven_eight = nump.zeros(len(dimensionality))
#
# for s in dimensionality:
#     for i in range(s + 1, 784):
#         distortion_four_seven_eight[dimensionality.index(s)] += nump.abs(eigenvalues_four_seven_eight[i])
#         distortion_four[dimensionality.index(s)] += nump.abs(eigenvalues_four[i])
#         distortion_seven[dimensionality.index(s)] += nump.abs(eigenvalues_seven[i])
#         distortion_eight[dimensionality.index(s)] += nump.abs(eigenvalues_eight[i])

# plot.plot(dimensionality, distortion_four)
# plot.xlabel("M")
# plot.ylabel("J")
# plot.title("Total Distortion Errors as a function of dimension M produced by PCA on Number 4")
# plot.show()
# plot.plot(dimensionality, distortion_seven)
# plot.xlabel("M")
# plot.ylabel("J")
# plot.title("Total Distortion Errors as a function of dimension M produced by PCA on Number 7")
# plot.show()
# plot.plot(dimensionality, distortion_eight)
# plot.xlabel("M")
# plot.ylabel("J")
# plot.title("Total Distortion Errors as a function of dimension M produced by PCA on Number 8")
# plot.show()
# plot.plot(dimensionality, distortion_four_seven_eight)
# plot.xlabel("M")
# plot.ylabel("J")
# plot.title("Total Distortion Errors as a function of dimension M produced by PCA on Number 4, 7 and 8")
# plot.show()


# -----to be fixed------
# for s in dimensionality:
#     projection_matrix_four = eigenvectors_four[0:s]
#     projection_matrix_seven = eigenvectors_seven[0:s]
#     projection_matrix_eight = eigenvectors_eight[0:s]
#     Z = nump.dot(projection_matrix_four, fours.T)
#
#     pro = nump.dot(projection_matrix_four.T, Z)
#     dis = nump.sum(nump.power(nump.linalg.norm(pro - fours.T, axis=1), 2))
#     distortion_four[dimensionality.index(s)] = dis / len(fours)

#     dis = 0
#     for i in range(len(fours)):
#         uu = nump.dot(projection_matrix_four, projection_matrix_four.T)
#         pro = nump.dot(uu, fours.T)
#         dis += nump.power(nump.linalg.norm(pro - fours.T[i]), 2)
#     distortion_four[dimensionality.index(s)] = dis / len(fours)
#
# plot.plot(dimensionality, distortion_four)
# plot.xlabel("M")
# plot.ylabel("J")
# plot.show()
# ------------------------------------------------------------------------
#

# !! The correct way to reconstruct the images from PCA.
# M = 300
# for i in range(len(fours)):
#     projection = nump.dot(eigenvectors_four[0:M].T, nump.dot(eigenvectors_four[0:M], fours[i].T)).reshape(28, 28)
#     plot.imshow(nump.abs(projection), cmap='gray')
#     plot.show()


# print("PCA finishes on number 4, 7 and 8.")
#
# print("LDA starts...")
#
# total_n = int(len(fours)) + int(len(sevens)) + int(len(eights))
# within_four_seven_eight = (cov_four * len(fours) + cov_seven * len(sevens) + cov_eight * len(eights)) / total_n
# cov_T_four = nump.outer(mean_four - mean_four_seven_eight, mean_four - mean_four_seven_eight)
# cov_T_seven = nump.outer(mean_seven - mean_four_seven_eight, mean_seven - mean_four_seven_eight)
# cov_T_eight = nump.outer(mean_eight - mean_four_seven_eight, mean_eight - mean_four_seven_eight)
# between_four_seven_eight = (cov_T_four * len(fours) + cov_T_seven * len(sevens) + cov_T_eight * len(eights)) / total_n
# U, X, V = nump.linalg.svd(nump.dot(nump.linalg.pinv(within_four_seven_eight), between_four_seven_eight))
#
# within_four_seven_eight_inverse = nump.linalg.pinv(within_four_seven_eight)
# eigenvalues_LDA, eigenvectors_LDA = \
#     nump.linalg.eig(nump.dot(within_four_seven_eight_inverse, between_four_seven_eight))
#
# sorted_within_four_seven_eight = eigenvalues_LDA.argsort()[::-1]
# eigenvalues_LDA = eigenvalues_LDA[sorted_within_four_seven_eight]
# eigenvectors_LDA = eigenvectors_LDA[sorted_within_four_seven_eight]
#
# print("The maximum dimension that LDA can achieve of classification is: ")
# print(nump.ndim(eigenvectors_LDA))
# print("This is the number of classes in total subtracted by 1. This is the rank of the projection matrix of LDA.")


# projection_fours = nump.array([projection[key] for (key, label) in enumerate(fours_sevens_eights_labels)
#                                if int(label) == 4])
# projection_sevens = nump.array([projection[key] for (key, label) in enumerate(fours_sevens_eights_labels)
#                                 if int(label) == 7])
# projection_eights = nump.array([projection[key] for (key, label) in enumerate(fours_sevens_eights_labels)
#                                 if int(label) == 8])


# print("LDA finishes.")
#
# print("PCA plotting...")
# xs = nump.zeros(len(fours_sevens_eights))
# ys = nump.zeros(len(fours_sevens_eights))
# for i in range(len(fours_sevens_eights)):
#     projected_2D = nump.abs(nump.dot(eigenvectors_four_seven_eight[0:2], fours_sevens_eights[i].T))
#     xs[i] = projected_2D[0]
#     ys[i] = projected_2D[1]
#
# xs_four = nump.array([xs[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 4])
# ys_four = nump.array([ys[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 4])
# xs_seven = nump.array([xs[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 7])
# ys_seven = nump.array([ys[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 7])
# xs_eight = nump.array([xs[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 8])
# ys_eight = nump.array([ys[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 8])
# plot.scatter(xs_four, ys_four, c='green')
# plot.scatter(xs_seven, ys_seven, c='blue')
# plot.scatter(xs_eight, ys_eight, c='red')
# plot.legend(["four", "seven", "eight"])
# plot.title("PCA Visual")
# plot.show()
# print("PCA plotting completes.")
# print("LDA plotting...")
#
# for i in range(len(fours_sevens_eights)):
#     LDA_projected_2D = nump.abs(nump.dot(eigenvectors_LDA[0:2], fours_sevens_eights[i].T))
#     xs[i] = LDA_projected_2D[0]
#     ys[i] = LDA_projected_2D[1]
#
# xs_four = nump.array([xs[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 4])
# ys_four = nump.array([ys[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 4])
# xs_seven = nump.array([xs[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 7])
# ys_seven = nump.array([ys[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 7])
# xs_eight = nump.array([xs[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 8])
# ys_eight = nump.array([ys[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 8])
# plot.scatter(xs_four, ys_four, c='green')
# plot.scatter(xs_seven, ys_seven, c='blue')
# plot.scatter(xs_eight, ys_eight, c='red')
# plot.legend(["four", "seven", "eight"])
# plot.title("LDA Visual")
# plot.show()
#
# print("LDA plotting finishes.")

# print("tSNE plotting...")
# projected_2D = tSNE.tsne(fours_sevens_eights, 2, D)
# xs['tsne-2d-one'] = projected_2D[:, 0]
# xs['tsne-2d-two'] = projected_2D[:, 1]
# plot.scatter(
#     x="tsne-2d-one", y="tsne-2d-two",
#     hue="y",
#     data=xs,
#     legend="full",
#     alpha=0.3
# )

# fours_sevens_eights = fours_sevens_eights[10000:200001]
# xs = nump.zeros(len(fours_sevens_eights))
# ys = nump.zeros(len(fours_sevens_eights))
# tSNE_projected_2D = tSNE.tsne(fours_sevens_eights, 2, D)
# for i in range(len(fours_sevens_eights)):
#     projected_2D = tSNE.tsne(fours_sevens_eights[i], 2, D)
#     xs[i] = projected_2D[0]
#     ys[i] = projected_2D[1]
# xs_four = nump.array([xs[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 4])
# ys_four = nump.array([ys[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 4])
# xs_seven = nump.array([xs[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 7])
# ys_seven = nump.array([ys[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 7])
# xs_eight = nump.array([xs[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 8])
# ys_eight = nump.array([ys[key] for (key, label) in enumerate(fours_sevens_eights_labels) if int(label) == 8])
# plot.scatter(xs_four, ys_four, c='green')
# plot.scatter(xs_seven, ys_seven, c='blue')
# plot.scatter(xs_eight, ys_eight, c='red')
# plot.show()
print("tSNE plotting finishes.")

#
# print("Linear Regression starts...")
# L = nump.random.rand()
# fun = nump.random.rand(10, D)
# loss = nump.zeros(len(images))
# x = nump.array(images)
# y = nump.array(labels)
# for i in range(1000):
#     fun = nump.linalg.lstsq(nump.dot(x.T, x) + L, nump.dot(x.T, y))
#     pred = nump.dot(fun, x)
#     loss[i] = nump.square(pred - y.T) / 2
#
# print(loss)

# print("Linear Regression starts running...")
# # Ms = [2, 10, 50, 100, 200, 300, 500, 784]
# Ms = [300]
# STr = nump.zeros(len(Ms))
# STe = nump.zeros(len(Ms))
# for M in Ms:
#     print("Dimension " + str(M) + "...")
#     print("Training on 0...")
#     W_zero = nump.dot(zeros[:, 0:M].T, zeros[:, 0:M])
#     W_zero = nump.linalg.pinv(W_zero)
#     W_zero = nump.dot(W_zero, zeros[:, 0:M].T)
#     W_zero = nump.dot(W_zero, nump.repeat(0, len(zeros)))
#
#     print("Training on 1...")
#     W_one = nump.dot(ones[:, 0:M].T, ones[:, 0:M])
#     W_one = nump.linalg.pinv(W_one)
#     W_one = nump.dot(W_one, ones[:, 0:M].T)
#     W_one = nump.dot(W_one, nump.repeat(1, len(ones)))
#
#     print("Training on 2...")
#     W_two = nump.dot(twos[:, 0:M].T, twos[:, 0:M])
#     W_two = nump.linalg.pinv(W_two)
#     W_two = nump.dot(W_two, twos[:, 0:M].T)
#     W_two = nump.dot(W_two, nump.repeat(2, len(twos)))
#
#     print("Training on 3...")
#     W_three = nump.dot(threes[:, 0:M].T, threes[:, 0:M])
#     W_three = nump.linalg.pinv(W_three)
#     W_three = nump.dot(W_three, threes[:, 0:M].T)
#     W_three = nump.dot(W_three, nump.repeat(3, len(threes)))
#
#     print("Training on 4...")
#     W_four = nump.dot(fours[:, 0:M].T, fours[:, 0:M])
#     W_four = nump.linalg.pinv(W_four)
#     W_four = nump.dot(W_four, fours[:, 0:M].T)
#     W_four = nump.dot(W_four, nump.repeat(4, len(fours)))
#
#     print("Training on 5...")
#     W_five = nump.dot(fives[:, 0:M].T, fives[:, 0:M])
#     W_five = nump.linalg.pinv(W_five)
#     W_five = nump.dot(W_five, fives[:, 0:M].T)
#     W_five = nump.dot(W_five, nump.repeat(5, len(fives)))
#
#     print("Training on 6...")
#     W_six = nump.dot(sixes[:, 0:M].T, sixes[:, 0:M])
#     W_six = nump.linalg.pinv(W_six)
#     W_six = nump.dot(W_six, sixes[:, 0:M].T)
#     W_six = nump.dot(W_six, nump.repeat(6, len(sixes)))
#
#     print("Training on 7...")
#     W_seven = nump.dot(sevens[:, 0:M].T, sevens[:, 0:M])
#     W_seven = nump.linalg.pinv(W_seven)
#     W_seven = nump.dot(W_seven, sevens[:, 0:M].T)
#     W_seven = nump.dot(W_seven, nump.repeat(7, len(sevens)))
#
#     print("Training on 8...")
#     W_eight = nump.dot(eights[:, 0:M].T, eights[:, 0:M])
#     W_eight = nump.linalg.pinv(W_eight)
#     W_eight = nump.dot(W_eight, eights[:, 0:M].T)
#     W_eight = nump.dot(W_eight, nump.repeat(8, len(eights)))
#
#     print("Training on 9...")
#     W_nine = nump.dot(nines[:, 0:M].T, nines[:, 0:M])
#     W_nine = nump.linalg.pinv(W_nine)
#     W_nine = nump.dot(W_nine, nines[:, 0:M].T)
#     W_nine = nump.dot(W_nine, nump.repeat(9, len(nines)))
#
#     print("Predicting on training images...")
#     d = nump.zeros(10)
#     Pr = nump.zeros(len(labels))
#     for i in range(len(images)):
#         p0 = nump.dot(images[i][0:M], W_zero)
#         p1 = nump.dot(images[i][0:M], W_one)
#         p2 = nump.dot(images[i][0:M], W_two)
#         p3 = nump.dot(images[i][0:M], W_three)
#         p4 = nump.dot(images[i][0:M], W_four)
#         p5 = nump.dot(images[i][0:M], W_five)
#         p6 = nump.dot(images[i][0:M], W_six)
#         p7 = nump.dot(images[i][0:M], W_seven)
#         p8 = nump.dot(images[i][0:M], W_eight)
#         p9 = nump.dot(images[i][0:M], W_nine)
#         y = labels[i]
#         d = [p0 - y, p1 - y, p2 - y, p3 - y, p4 - y, p5 - y, p6 - y, p7 - y, p8 - y, p9 - y]
#         d = nump.abs(d)
#         Pr[i] = nump.where(d == min(d))[0][0]
#
#     print("Evaluating the training predictions...")
#     C = nump.zeros((10, 2), dtype=int)
#     for i in range(len(labels)):
#         if Pr[i] == labels[i]:
#             C[labels[i]][0] += 1
#         else:
#             C[labels[i]][1] -= 1
#     print(C)
#     STr[Ms.index(M)] = nump.sum(C.T[0]) / (nump.sum(C.T[0]) - nump.sum(C.T[1]))
#     print("Average success rate: " + str(STr[Ms.index(M)]))
#     print("Predicting on test images...")
#     d = nump.zeros(10)
#     Pr = nump.zeros(len(test_labels))
#     for i in range(len(test_images)):
#         p0 = nump.dot(test_images[i][0:M], W_zero)
#         p1 = nump.dot(test_images[i][0:M], W_one)
#         p2 = nump.dot(test_images[i][0:M], W_two)
#         p3 = nump.dot(test_images[i][0:M], W_three)
#         p4 = nump.dot(test_images[i][0:M], W_four)
#         p5 = nump.dot(test_images[i][0:M], W_five)
#         p6 = nump.dot(test_images[i][0:M], W_six)
#         p7 = nump.dot(test_images[i][0:M], W_seven)
#         p8 = nump.dot(test_images[i][0:M], W_eight)
#         p9 = nump.dot(test_images[i][0:M], W_nine)
#         y = test_labels[i]
#         d = [p0 - y, p1 - y, p2 - y, p3 - y, p4 - y, p5 - y, p6 - y, p7 - y, p8 - y, p9 - y]
#         d = nump.abs(d)
#         Pr[i] = nump.where(d == min(d))[0][0]
#
#     print("Evaluating the testing predictions...")
#     C = nump.zeros((10, 2), dtype=int)
#     for i in range(len(test_labels)):
#         if Pr[i] == test_labels[i]:
#             C[test_labels[i]][0] += 1
#         else:
#             C[test_labels[i]][1] -= 1
#
#     print(C)
#     STe[Ms.index(M)] = nump.sum(C.T[0]) / (nump.sum(C.T[0]) - nump.sum(C.T[1]))
#     print("Average success rate: " + str(STe[Ms.index(M)]))
#
# plot.plot(Ms, STr)
# plot.plot(Ms, STe)
# plot.legend(['success rate for training images', 'success rate for testing images'])
# plot.title('Success Rate for Training and Testing Images')
# plot.show()
# print("Linear Regression completes.")

print("Logistic Regression starts...")
ones_twos = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 1 or int(label) == 2])
ones_twos_label = nump.array([labels[key] for (key, label) in enumerate(labels) if int(label) == 1 or int(label) == 2])
ones_twos_test = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if int(label) == 1
                             or int(label) == 2])
ones_twos_test_label = nump.array([test_labels[key] for (key, label) in enumerate(test_labels) if int(label) == 1
                                  or int(label) == 2])
M = 200
Y = nump.array(ones_twos_label)
W = nump.random.rand(M)
SIGMA = 1
L = 2
X = nump.array(ones_twos)
X = X[:, 0:M]


def objective_function(w):
    e = 0
    for i in range(len(ones_twos)):
        e += 1 / (1 + nump.exp((-1) * SIGMA * Y[i] * nump.dot(X[i], w.T)))
    return e


def gradient(w):
    de = 0
    for i in range(len(ones_twos)):
        sig = 1 / (1 + nump.exp((-1) * SIGMA * Y[i] * nump.dot(X[i], w.T)))
        de += sig * (1 - sig) * Y[i] * X[i]
    return de


print("Gradient descending...")
count = 0
Error = 100000
LastError = 100000
count = 0
while Error > 1:
    count += 1
    print(count)
    LastError = Error
    Error = objective_function(W)
    print(Error)
    W = W - L * gradient(W)

print(W)
print("Testing...")
success = 0
failure = 0
for i in range(len(ones_twos_test_label)):
    P = nump.max(nump.dot(ones_twos_test[i][0:M], W.T))
    if P == ones_twos_test_label[i]:
        success += 1
    else:
        failure += 1

print(len(ones_twos_test_label))
print(success)
print(failure)

print("Logistic Regression completes.")
