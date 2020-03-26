import numpy as nump
import matplotlib.pyplot as plot
import struct as struc
# import gzip
# with gzip.open('../MNIST/t10k-images-idx3-ubyte.gz', 'rb') as f:
#     test_images = f.read()
# with gzip.open('../MNIST/t10k-labels-idx1-ubyte.gz', 'rb') as f:
#     test_labels = f.read()
# with gzip.open('../MNIST/train-images-idx3-ubyte.gz', 'rb') as f:
#     train_images = f.read()
# with gzip.open('../MNIST/train-labels-idx1-ubyte.gz', 'rb') as f:
#     train_labels = f.read()

filename = {'images': '../MNIST/train-images-idx3-ubyte.gz', 'labels': 'train-labels.idx1-ubyte'}
train_images = open(filename['images'], 'rb')
train_images.seek(0)
magic = struc.unpack('>4B', train_images.read(4))
n = struc.unpack('>I', train_images.read(4))[0]
row = struc.unpack('>I', train_images.read(4))[0]
col = struc.unpack('>I', train_images.read(4))[0]
total_bytes = n * row * col
images = 255 - \
         nump.asarray(struc.unpack('>' + 'B' * total_bytes, train_images.read(total_bytes))).reshape((n, row, col))


print(n)
print(row)
print(col)
print(images)
