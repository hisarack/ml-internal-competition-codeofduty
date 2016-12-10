from keras.models import Sequential
from keras.layers import MaxoutDense, Dense, Dropout, Activation
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop, Adam, Adamax

# for CNN
from keras.layers import Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

import numpy as np
import sys

first_tag = True

raw = []

with open("./train2.csv") as f:
    for line in f:
        tmp = line.strip().split(",")
        label = tmp[0]
        digit = tmp[1:]
        if first_tag is True:
            first_tag = False
        else:
            raw.append((label, digit))
f.close()


def reslplit():
    x0 = []
    x1 = []
    y0 = []
    y1 = []
    np.random.shuffle(raw)

    for label, digit in raw:
        if len(x1) < 2000:
            x1.append(digit)
            y1.append(label)
        else:
            x0.append(digit)
            y0.append(label)

    x0 = np.array(x0)
    y0 = np.array(y0)
    x1 = np.array(x1)
    y1 = np.array(y1)

    x0 = x0.astype('float32')
    x1 = x1.astype('float32')

    x0 /= 255
    x1 /= 255

    print x0.shape
    print y0.shape

    print x1.shape
    print y1.shape
    return x0, x1, y0, y1


def cnn(x0, y0, x1, y1, nb_epoch, optimizer):

    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(1, 28, 28)))
    model.add(Convolution2D(32, 5, 5, border_mode='same', input_shape=(1, 30, 30)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 5, 5, border_mode='same'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(32, 5, 5))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    # sgd = SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(
        loss='categorical_crossentropy',
        optimizer=optimizer,
        metrics=["accuracy"]
    )

    x0 = x0.reshape(x0.shape[0], 1, 28, 28)
    x1 = x1.reshape(x1.shape[0], 1, 28, 28)

    y0 = np_utils.to_categorical(y0, 10)
    y1 = np_utils.to_categorical(y1, 10)

    model.fit(
        x0, y0,
        nb_epoch=nb_epoch,
        batch_size=256,
        validation_data=(x1, y1)
    )
    result = model.evaluate(x1, y1, batch_size=256)
    return model, result


while True:
    index = 0
    nb_epoch = 50
    adamax_opt = Adamax(lr=0.008)
    x0, x1, y0, y1 = reslplit()
    name = "/usr/share/hisarack-codeofduty/keras/models/{}_{}.h5".format(sys.argv[1], index)
    model, result = cnn(x0, y0, x1, y1, nb_epoch, adamax_opt)
    if result[0] < 0.03:
        print "#######################"
        print result
        print "#######################"
        model.save_weights(name)
        index = index+1
