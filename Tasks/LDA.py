from mnist import MNIST
import numpy as nump

print("Loading the data...")

data = MNIST("../data/MNIST/")

images, labels = data.load_training()
print("LDA starts...")

fours = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 4])
sevens = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 7])
eights = nump.array([images[key] for (key, label) in enumerate(labels) if int(label) == 8])

cov_four = nump.cov(fours.T)
cov_seven = nump.cov(sevens.T)
cov_eight = nump.cov(eights.T)

within_four_seven = nump.add(cov_four, cov_seven)
between_four_seven = nump.matmul(nump.subtract(cov_four, cov_seven), (nump.subtract(cov_four, cov_seven)).T)
projection_matrix_LDA = nump.linalg.pinv(within_four_seven)
projection_matrix_LDA = nump.matmul(projection_matrix_LDA, between_four_seven)
eigenvalues_LDA, eigenvectors_LDA = nump.linalg.eig(projection_matrix_LDA)
sorted_LDA = eigenvectors_LDA.argsort()[::-1]
eigenvalues_LDA = eigenvalues_LDA[sorted_LDA]
eigenvectors_LDA = eigenvectors_LDA[0:20]
eigenvectors_LDA = eigenvectors_LDA[sorted_LDA]
print(eigenvalues_LDA)
print(eigenvectors_LDA)
