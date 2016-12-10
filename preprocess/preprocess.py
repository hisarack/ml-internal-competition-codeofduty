import numpy as np
import numpy.matlib
import hexdump
import mnist_helpers
from sklearn.decomposition import PCA

from scipy.ndimage.interpolation import map_coordinates
from scipy.ndimage.filters import gaussian_filter


def extract_img(img_fname):
    with open(img_fname, 'r') as f:
        buf = f.read()
        digits = []
        for d in hexdump.dump(buf).split(" "):
            digits.append(str(int(d, 16)))
        return digits


def transform_dataset(source_fname, target_fname, func):

    tf = open(target_fname, 'w+')

    if func == "train":
        with open(source_fname, 'r') as sf:
            for line in sf:
                img_fname, img_label = line.rstrip('\n').split(",")
                img_buffer = extract_img('./train/{}'.format(img_fname))
                if len(img_buffer) != 784:
                    print "error!"
                tf.write('{},{}\n'.format(img_label, ','.join(img_buffer)))
            tf.close()
    elif func == "test":
        with open(source_fname, 'r') as sf:
            for line in sf:
                img_fname = line.rstrip('\n')
                img_buffer = extract_img('./test/{}'.format(img_fname))
                if len(img_buffer) != 784:
                    print "error!"
                tf.write('{},{}\n'.format(img_fname, ','.join(img_buffer)))
            tf.close()


def elastic_transform(source_fname, target_fname, func):

    def _elastic_transform(img_buffer, alpha=6, sigma=6):
        random_state = np.random.RandomState(None)
        shape = img_buffer.shape
        dx = gaussian_filter((random_state.rand(*shape)*2-1), sigma, mode="constant", cval=0)*alpha
        dy = gaussian_filter((random_state.rand(*shape)*2-1), sigma, mode="constant", cval=0)*alpha

        x, y = np.meshgrid(np.arange(shape[0]), np.arange(shape[1]))
        indices = np.reshape(y+dy, (-1, 1)), np.reshape(x+dx, (-1, 1))

        return map_coordinates(img_buffer, indices, order=1).reshape(shape)

    tf = open(target_fname, 'w+')

    if func == "train":
        with open(source_fname, 'r') as sf:
            for line in sf:
                img_label  = line.split(",")[0]
                img_buffer = line.split(",")[1:]

                img_buffer = np.array(img_buffer)
                img_buffer = img_buffer.astype(int)
                img_buffer = np.reshape(img_buffer, (28, 28))
                img_buffer = _elastic_transform(img_buffer)
                # img_buffer = mnist_helpers.do_cropping(img_buffer)
                img_buffer = np.reshape(img_buffer, (784)).astype(int)
                img_buffer = img_buffer.astype(str)

                tf.write('{},{}\n'.format(img_label, ','.join(img_buffer)))
            tf.close()
    elif func == "test":
        with open(source_fname, 'r') as sf:
            for line in sf:
                img_fname  = line.split(",")[0]
                img_buffer = line.split(",")[1:]

                img_buffer = np.array(img_buffer)
                img_buffer = img_buffer.astype(int)
                img_buffer = np.reshape(img_buffer, (28, 28))
                img_buffer = _elastic_transform(img_buffer)
                # img_buffer = mnist_helpers.do_cropping(img_buffer)
                img_buffer = np.reshape(img_buffer, (784)).astype(int)
                img_buffer = img_buffer.astype(str)

                tf.write('{},{}\n'.format(img_fname, ','.join(img_buffer)))
            tf.close()


class ZCA:

    def __init__(self, bias=0.1, copy=False):
        self.bias = bias
        self.copy = copy

    def fit(self, X, y=None):
        # n_features, n_samples = X.shape
        # print X.shape
        X = X.T
        self.mean_ = np.mean(X, axis=1)
        X -= np.tile(self.mean_, (X.shape[1], 1)).T
        sigma = X.dot(X.T)/X.shape[1]
        U, S, V = np.linalg.svd(sigma)
        self.components_ = U.dot(np.diag(1.0 / (S + self.bias))).dot(U.T)
        return self

    def transform(self, X):
        X = X.copy()
        X -= self.mean_
        X_transformed = np.dot(self.components_, X)
        # X_transformed += self.mean_
        return X_transformed

    def inverse_transform(self, X_transformed):
        X = X_transformed.copy()
        X = np.dot(self.components_, X)
        X += self.mean_
        return X


def zca_whiten_fit(source_fname):

    with open(source_fname, 'r') as sf:
        img_buffer_list = []
        for line in sf:
            img_buffer = line.split(",")[1:]
            img_buffer = np.array(img_buffer)
            img_buffer = img_buffer.astype('float32')
            img_buffer = np.reshape(img_buffer, (784))
            img_buffer_list.append(img_buffer)

        X = np.array(img_buffer_list)
        X /= 255
        zca = ZCA()
        zca = zca.fit(X)
        return zca


def zca_whiten_transform(source_fname, target_fname, zca):

    tf = open(target_fname, 'w+')

    with open(source_fname, 'r') as sf:
        for line in sf:
            img_label  = line.split(",")[0]
            img_buffer = line.split(",")[1:]

            img_buffer = np.array(img_buffer).astype('float32')
            img_buffer /= 255
            img_buffer = zca.transform(img_buffer)
            img_buffer *= 255
            img_buffer = img_buffer.astype(int)

            img_buffer = img_buffer.clip(min=0, max=255)
            low_idxs = img_buffer < 128
            img_buffer[low_idxs] = 0

            img_buffer = np.reshape(img_buffer, (784))
            img_buffer = img_buffer.astype(str)
            tf.write('{},{}\n'.format(img_label, ','.join(img_buffer)))
        tf.close()


def pca_whiten_fit(source_fname):

    with open(source_fname, 'r') as sf:
        img_buffer_list = []
        for line in sf:
            img_buffer = line.split(",")[1:]
            img_buffer = np.array(img_buffer)
            img_buffer = img_buffer.astype('float32')
            img_buffer = np.reshape(img_buffer, (784))
            img_buffer_list.append(img_buffer)

        X = np.array(img_buffer_list)
        pca = PCA(whiten=True, copy=True)
        pca = pca.fit(X)
        return pca


def pca_whiten_transform(source_fname, target_fname, pca):

    tf = open(target_fname, 'w+')

    with open(source_fname, 'r') as sf:
        for line in sf:
            img_label  = line.split(",")[0]
            img_buffer = line.split(",")[1:]

            img_buffer = np.array(img_buffer).astype('float32')
            img_buffer = pca.inverse_transform(pca.transform(img_buffer))
            img_buffer = img_buffer.astype(int)
            # img_buffer = zca.transform(img_buffer)
            # clim = np.max(np.abs(img_buffer))
            # img_buffer = np.reshape(img_buffer, (784))
            # img_buffer -= np.average(img_buffer)
            # img_buffer /= clim

            img_buffer = img_buffer.clip(min=0)
            img_buffer.flatten()
            img_buffer = img_buffer.astype(str)
            img_buffer = img_buffer.tolist()[0]
            tf.write('{},{}\n'.format(img_label, ','.join(img_buffer)))
        tf.close()


# transform_dataset('./train.csv', './train2.csv', "train")
transform_dataset('./testnew.csv', './testnew2.csv', "test")

# elastic_transform("./train2.csv", "./train3.csv" , "train")
# elastic_transform("./test2.csv", "./test3.csv" , "train")

# zca = zca_whiten_fit("./train2.csv")
# zca_whiten_transform("./train2.csv", "./train4.csv", zca)
# zca_whiten_transform("./test2.csv", "./test4.csv", zca)
