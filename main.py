from keras.models import Sequential
from keras.layers import MaxoutDense, Dense, Dropout, Activation
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop, Adam, Adamax

# for CNN
from keras.layers import Flatten
from keras.layers import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.utils import np_utils
from keras.callbacks import ModelCheckpoint

import numpy as np

x0 = []
y0 = []
x1 = []
y1 = []

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

mean_ = np.mean(x0, axis=0)
print mean_.shape

x0 -= np.tile(mean_, (x0.shape[0], 1))
x1 -= np.tile(mean_, (x1.shape[0], 1))

print x0.shape
print y0.shape

print x1.shape
print y1.shape


def mlp1(x0, y0, x1, y1):

    model = Sequential()

    model.add(Dense(input_dim=784, output_dim=512, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=512,  output_dim=512, init='uniform'))
    model.add(Activation('tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=512, output_dim=10, init='uniform'))
    model.add(Activation('softmax'))

    # sgd = SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)
    rms = RMSprop()

    y0 = np_utils.to_categorical(y0, 10)
    y1 = np_utils.to_categorical(y1, 10)

    model.compile(loss='categorical_crossentropy', optimizer=rms)
    model.fit(x0, y0,
              nb_epoch=30,
              batch_size=128,
              show_accuracy=True,
              verbose=2,
              validation_data=(x1, y1))
    result = model.evaluate(x1, y1, batch_size=128)
    return model, result


def mlp2(x0, y0, x1, y1, nb_epoch):

    model = Sequential()

    model.add(Dense(input_dim=784, output_dim=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=512,  output_dim=512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(input_dim=512, output_dim=10, activation='softmax'))

    # sgd = SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)
    rms = RMSprop()

    y0 = np_utils.to_categorical(y0, 10)
    y1 = np_utils.to_categorical(y1, 10)

    model.compile(loss='categorical_crossentropy', optimizer=rms)
    model.fit(x0, y0,
              nb_epoch=nb_epoch,
              batch_size=128,
              show_accuracy=True,
              verbose=2,
              validation_data=(x1, y1))
    print model.evaluate(x1, y1, batch_size=128)
    return model


def cnn3(x0, y0, x1, y1, nb_epoch, optimizer):


    model = Sequential()

    model.add(ZeroPadding2D((1, 1), input_shape=(1, 28, 28)))
    model.add(Convolution2D(64, 3, 3, border_mode='same', input_shape=(1, 30, 30)))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3, border_mode='same'))
    model.add(Activation('relu'))
    model.add(ZeroPadding2D((1, 1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(1024))
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

    model.fit(x0, y0, nb_epoch=nb_epoch, batch_size=256)
    print model.evaluate(x1, y1, batch_size=256)
    return model




def cnn2(x0, y0, x1, y1, nb_epoch, optimizer):

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


def cnn(x0, y0, x1, y1, nb_epoch, optimizer):

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, 28, 28)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))

    model.add(Dense(10))
    model.add(Activation('softmax'))

    # sgd = SGD(lr=0.5, decay=1e-6, momentum=0.9, nesterov=True)

    model.compile(loss='categorical_crossentropy', optimizer=optimizer)

    x0 = x0.reshape(x0.shape[0], 1, 28, 28)
    x1 = x1.reshape(x1.shape[0], 1, 28, 28)

    y0 = np_utils.to_categorical(y0, 10)
    y1 = np_utils.to_categorical(y1, 10)

    model.fit(
        x0, y0,
        nb_epoch=nb_epoch,
        batch_size=128,
        validation_data=(x1, y1)
    )

    return model


def predict(model, fname, is_cnn=False):

    x2 = []
    id2 = []

    with open("./test2.csv") as f:
        for line in f:
            tmp = line.strip().split(",")
            dname = tmp[0]
            digit = tmp[1:]
            id2.append(dname)
            x2.append(digit)
    f.close()

    x2 = np.array(x2)

    x2 = x2.astype('float32')

    if is_cnn is True:
        x2 = x2.reshape(x2.shape[0], 1, 28, 28)
        x2 /= 255
        x2 -= np.tile(mean_, (x2.shape[0], 1))

    r2 = model.predict(x2)

    wf = open("./{}.csv".format(fname), 'w')
    # wf.write("ImageId,Label\n")
    imgId = 0
    for rs in r2:
        imgProb = []
        for r in rs:
            imgProb.append(str(r))
        if "1.0" in imgProb:
            index = imgProb.index("1.0")
            imgProb = ["0.0"] * 10
            imgProb[index] = "1.0"
        wf.write(id2[imgId]+","+",".join(imgProb)+"\n")
        imgId = imgId + 1
    wf.close()

find_good = False
while find_good is False:
    nb_epoch = 200
    adamax_opt = Adamax(lr=0.008)
    model, result = cnn2(x0, y0, x1, y1, nb_epoch, adamax_opt)
    if result[0] < 0.005:
        print "#######################"
        print result[0]
        print "#######################"
        predict(model, "{}_{}".format("cnn_zca_adamax", nb_epoch), True)
        find_good = True
