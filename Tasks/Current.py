from mnist import MNIST
import numpy as nump
from sklearn.manifold import TSNE
import sys
import matplotlib.pyplot as plot

nump.set_printoptions(threshold=sys.maxsize)

print("Loading the data...")

data = MNIST("../data/MNIST/")

images, labels = data.load_training()
test_images, test_labels = data.load_testing()
print(nump.array(test_labels))

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

ones_twos_threes_training = nump.array([images[key] for (key, label) in enumerate(labels) if
                                        int(label) == 1 or int(label) == 2 or int(label) == 3])

ones_twos_threes_training_label = nump.array([labels[key] for (key, label) in enumerate(labels) if
                                              int(label) == 1 or int(label) == 2 or int(label) == 3])

ones_twos_threes_testing = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if
                                       int(label) == 1 or int(label) == 2 or int(label) == 3])

ones_twos_threes_testing_label = nump.array([test_labels[key] for (key, label) in enumerate(test_labels) if
                                             int(label) == 1 or int(label) == 2 or int(label) == 3])


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
# distortion_four_seven_eight = nump.zeros(len(dimensionality))
#
# for s in dimensionality:
#     for i in range(s + 1, 784):
#         distortion_four_seven_eight[dimensionality.index(s)] += nump.abs(eigenvalues_four_seven_eight[i])

# plot.plot(dimensionality, distortion_four_seven_eight)
# plot.xlabel("M")
# plot.ylabel("J")
# plot.title("Total Distortion Errors as a function of dimension M produced by PCA on Number 4, 7 and 8")
# plot.show()

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
#
# print("t-SNE running...")
# print("This might take several minutes...")
# X = fours_sevens_eights[:, 0:300]
# Y = fours_sevens_eights_labels
# X_embedded = TSNE(n_components=2, random_state=0, n_iter=260).fit_transform(X)
# ls = 0, 4, 7, 8
# plot.figure()
# colors = 'w', 'g', 'b', 'r'
# for i, c, l in zip(fours_sevens_eights_labels, colors, ls):
#     plot.scatter(X_embedded[Y == i, 0], X_embedded[Y == i, 1], c=c, label=l)
# plot.legend()
# plot.show()

# print("t-SNE plotting finishes.")


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
# Ms = [500]
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

# print("Logistic Regression starts...")

#
# def sigmoid(w, xl):
#     return 1 / (1 + nump.exp(SIGMA * nump.dot(xl, w.T)))


# def gradient(m, k, w, it):
#     g = nump.zeros((C, m))
#     sample_index = nump.random.randint(0, len(images))
#     probability = sigmoid(w[k], X[sample_index])
#     if Y[sample_index] == k:
#         indicator = 1
#     else:
#         indicator = 0
#     g[k] = -X[sample_index] * (indicator - probability)
#     Loss[M.index(m)][it] += sigmoid(w[k], X[sample_index])
#     return g


# Learning Rate
# L = 0.5

# Feature Dimension
# M = [2, 50, 100, 200, 300, 500, 784]
# Due to the large value of the data, if the magnitude of SIGMA is larger by even one factor of 10 than
# this then it will cause a RuntimeOverflow warning.
SIGMA = -0.00001
tests = nump.array([test_images[key] for (key, label) in enumerate(test_labels)])
# XX = nump.array(images)
# YY = nump.array(labels)
# S = nump.zeros(len(M))
# F = nump.zeros(len(M))
# R = nump.zeros(len(M))
# IT = 100000
# Loss = nump.zeros((len(M), IT))
# for m in M:
#     X = XX[:, 0:m]
#     Y = YY
#     N = len(Y)
#     TX = nump.array(tests)
#     TX = TX[:, 0:m]
#     TY = nump.array(test_labels)
#     TN = len(TX)
#     C = 10
#     W = nump.random.rand(C, m)
#     print("Stochastic gradient descending for dimension = " + str(m) + " ...")
#     for t in range(IT):
#         for k in range(C):
#             W = W - L * gradient(m, k, W, t)
#     print("Testing for dimension = " + str(m) + "...")
#     for i in range(TN):
#         P = nump.zeros(C)
#         Prediction = 0
#         maxP = 0
#         for k in range(C):
#             Probability = sigmoid(W[k], TX[i])
#             P[k] = Probability
#             if P[k] > maxP:
#                 maxP = P[k]
#                 Prediction = k
#         if Prediction == TY[i]:
#             S[M.index(m)] += 1
#         else:
#             F[M.index(m)] += 1
#         R[M.index(m)] = S[M.index(m)] / (S[M.index(m)] + F[M.index(m)])
# plot.plot(M, R)
# plot.title('Success Rate for Held-out Testing Images')
# plot.xlabel("M")
# plot.ylabel("Success Rate")
# plot.show()
#
# print("Logistic Regression completes.")

# M = 784
# # TX = nump.array(test_images)
# # TX = TX[:, 0:M]
# K = 10   # Total number of classes.
C = nump.exp(700)   # Penalty for misclassification.
# T = 100000  # Number of iterations.
# L = 0.00006     # Learning rate.
# TC = range(K)   # Total classes.
#
# print("Linear Soft-SVM starts...")
#
#
# def subgradient(w, img, lab, j):
#     sample_index = nump.random.randint(0, len(img))
#     sample_x = img[sample_index]
#     sample_y = lab[sample_index]
#     if j == sample_y:
#         sample_y = -1
#     else:
#         sample_y = 1
#     if 1 - sample_y * nump.dot(sample_x, w.T) < 0:
#         return w
#     else:
#         return w - C * sample_y * sample_x
#
#
# W = nump.random.rand(K, K, M)
# for i in range(K):
#     for j in range(i + 1, K):
#         Image = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == i or int(label) == j])
#         Label = nump.array([labels[key] for (key, label) in enumerate(labels) if int(label) == i or int(label) == j])
#         Image = Image[:, 0:M]
#         for it in range(T):
#             W[i][j] = W[i][j] - L * subgradient(W[i][j], Image, Label, j)
#
# S = 0
# F = 0
# A = 0
# TX = nump.array([test_images[key] for (key, label) in enumerate(test_labels) if int(label) in TC])
# TX = TX[:, 0:M]
# TL = nump.array([test_labels[key] for (key, label) in enumerate(test_labels) if int(label) in TC])
# for r in range(len(TL)):
#     Votes = nump.zeros(K)
#     for i in range(K):
#         for j in range(i + 1, K):
#             p = nump.dot(TX[r], W[i][j].T)
#             if p > 0:
#                 Votes[i] += 1
#             if p < 0:
#                 Votes[j] += 1
#     P = nump.where(Votes == nump.max(Votes))
#     print(Votes)
#     print(TL[r])
#     if len(P[0]) == 1 and TL[r] == P[0][0]:
#         S += 1
#     elif len(P[0]) == 1:
#         F += 1
#     else:
#         A += 1
#
# print(S)
# print(F)
# print(A)
#
# print("Linear Soft-SVM completes.")

# print("Non-Linear Gaussian RBF Kernel SVM starts...")
#
# X = nump.array(images)
# X = X[0:2000, :]
# Alpha = nump.random.rand(len(X))
# EPOCH = 10
# L = 0.00001
#
# print("Constructing kernel and training...")
#
# COV = nump.var(X)
#
#
# def rbf_kernel():
#     phi = nump.zeros((len(X), len(X)))
#     for i in range(len(X)):
#         for j in range(len(X)):
#             phi[i][j] = nump.exp(-0.5 * nump.power(nump.linalg.norm(X[i] - X[j]) / COV, 2))
#     return phi
#
#
# def gradient_boundary():
#     q = nump.zeros((len(X), len(X)))
#     for i in range(len(X)):
#         for j in range(len(X)):
#             q[i][j] = labels[i] * labels[j] * kernel[i][j]
#     return nump.dot(q, Alpha) - ones
#
#
# kernel = rbf_kernel()
# ones = nump.ones(len(X))
#
# for e in range(EPOCH):
#     Alpha = Alpha - L * gradient_boundary()
#
# print("Tesing...")
#
# TX = nump.array(test_images)
# for i in range(len(TX)):
#     f = 0
#     for n in range(len(X)):
#         f += Alpha[n] * labels[n] * nump.exp(-0.5 * COV * nump.power(nump.linalg.norm(X[n] - TX[i]), 2))
#     print(f)
#
# print("Non-Linear Gaussian RBF Kernel SVM finishes.")

print("FC Neural Networks starts...")

print("Training NN...")

M = 784


def acti(w, a):
    return (1 / (1 + nump.exp(SIGMA * nump.dot(a, w)))).reshape(len(w[0]), 1)


S = 0
F = 0
T = 100000
L = 0.0000000005
SIGMA = 0.001
nump.set_printoptions(precision=20)
N1 = 20
# N2 = 60
Layer1 = nump.ones(N1).reshape(N1, 1)
# Layer2 = nump.ones(N2).reshape(N2, 1)
Weights1 = nump.random.rand(M, len(Layer1) - 1)
Weights2 = nump.random.rand(len(Layer1), 3)
# Weights3 = nump.random.rand(len(Layer2), 10)
inputs = nump.array(fours_sevens_eights)
inputs = inputs[:, 0:M]
Error = nump.zeros((3, 1))
for it in range(T):
    Weights1_Partial = nump.zeros((M, len(Layer1) - 1))
    Weights2_Partial = nump.zeros((len(Layer1), 3))
    ind = nump.random.randint(0, len(inputs))
    x = inputs[ind]
    Layer0 = x
    # Feed forward
    Layer1[1:] = acti(Weights1, Layer0)
    # Layer1[1:] = nump.dot(Layer0.T, Weights1).reshape(19, 1)
    # Layer1[1:] = nump.exp(Layer1[1:]) / nump.sum(nump.exp(Layer1[1:]))
    Output = nump.dot(Layer1.T, Weights2)
    Output = nump.exp(SIGMA * Output) / nump.sum(nump.exp(SIGMA * Output))
    prediction = 0
    if fours_sevens_eights_labels[ind] == 4 and Output[0][0] == nump.max(Output):
        Error[0] = 1 - Output[0][0]
        Error[1] = Output[0][1]
        Error[2] = Output[0][2]
    if fours_sevens_eights_labels[ind] == 8 and Output[0][1] == nump.max(Output):
        Error[0] = Output[0][0]
        Error[1] = 1 - Output[0][1]
        Error[2] = Output[0][2]
    if fours_sevens_eights_labels[ind] == 9 and Output[0][1] == nump.max(Output):
        Error[0] = Output[0][0]
        Error[1] = Output[0][1]
        Error[2] = 1 - Output[0][2]
    # for k in range(len(Output)):
    #     Error += -Output[k] * nump.log(labels[ind]) - (1 - Output[k]) * nump.log(1 - labels[ind])

    # Back propagate
    Layer1_Error = Layer1[1:] * (1 - Layer1[1:]) * nump.dot(Weights2[1:], Error)
    # partial derivatives
    for k in range(len(Layer1) - 1):
        Weights1_Partial[:, k] = Layer1_Error[k] * Layer0
    for k in range(len(Output) - 1):
        Weights2_Partial[:, k] = (Error[k] * Layer1).reshape(N1)
    # Update
    Weights1 = Weights1 - L * Weights1_Partial
    Weights2 = Weights2 - L * Weights2_Partial

    # print(Weights1)
    # print(Weights2)

    # for i in range(len(inputs)):
    #     Layer0 = inputs[i]
    #     # Feed forward
    #     Layer1[1:] = acti(Weights1, Layer0)
    #     # Layer2[1:] = acti(Weights2, Layer1)
    #     Output = acti(Weights2, Layer1)
    #     # Back propagate
    #     Error[labels[i]] = abs(Output[labels[i]] - labels[i])
    #     # Layer2_Error += Layer2[1:] * (1 - Layer2[1:]) * nump.dot(Weights3[1:], Error)
    #     # Layer1_Error += Layer1[1:] * (1 - Layer1[1:]) * nump.dot(Weights2[1:], Layer2_Error)
    #     Layer1_Error = Layer1[1:] * (1 - Layer1[1:]) * nump.dot(Weights2[1:], Error)
    #     # partial derivatives
    #     for k in range(len(Layer1) - 1):
    #         Weights1_Partial[:, k] = Layer1_Error[k] * Layer0
    #     for k in range(len(Output) - 1):
    #         Weights2_Partial[:, k] = (Error[k] * Layer1).reshape(N1)
    #     # for k in range(len(Output)):
    #         # Weights3_Partial = Error[k] * Layer2
    #     # Update
    #     Weights1 = Weights1 - L * Weights1_Partial
    #     Weights2 = Weights2 - L * Weights2_Partial
    #     # Weights3 = Weights3 - L * Weights3_Partial

print("Testing NN...")
testing_inputs = nump.array(fours_sevens_eights_test)
testing_inputs = testing_inputs[:, 0:M]
for i in range(len(fours_sevens_eights_test)):
    Layer0 = testing_inputs[i]
    Layer1[1:] = acti(Weights1, Layer0)
    # Layer1[1:] = nump.dot(Layer0.T, Weights1).reshape(19, 1)
    # Layer1[1:] = nump.exp(Layer1[1:]) / nump.sum(nump.exp(Layer1[1:]))
    # print(Layer1[1:])
    # Layer2[1:] = acti(Weights2, Layer1)
    Output = nump.dot(Layer1.T, Weights2)
    Output = nump.exp(SIGMA * Output) / nump.sum(nump.exp(SIGMA * Output))
    if nump.where(Output == nump.max(Output))[0][0] == fours_sevens_eights_labels[i]:
        S += 1
    else:
        F += 1
    # Output = -1 * nump.log(1 - Output) + nump.log(Output)
    # Output = acti(Weights3, Layer2)
    # print("----------------------------------------------------------------------")
    # print(Output)
    # print(fours_sevens_eights_test_labels[i])
    # print(nump.where(Output == nump.max(Output))[0])

    # if (Output >= 0.5 and zeros_ones_testing_label[i] == 1) or (output < 0.5 and zeros_ones_testing_label[i] == 0):
    #     S += 1
    # else:
    #     F += 1

print(S)
print(F)

print("FC Neural Networks ends.")
