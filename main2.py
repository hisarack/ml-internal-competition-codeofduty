from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.optimizers import SGD, Adadelta, Adagrad, RMSprop, Adam, Adamax

# for CNN
from keras.layers import Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.utils import np_utils

import numpy as np

x0 = []
y0 = []
x1 = []
y1 = []

first_tag = True

with open("./train.csv") as f:
    for line in f:
        tmp = line.strip().split(",")
        label = tmp[0]
        digit = tmp[1:]
        if first_tag is True:
            first_tag = False
        elif len(x0) < 20000:
            x0.append(digit)
            y0.append(label)
        else:
            x1.append(digit)
            y1.append(label)
f.close()

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



def cnn(x0, y0, x1, y1, nb_epoch, optimizer):

    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode='valid', input_shape=(1, 28, 28)))
    model.add(Activation('relu'))
    model.add(Convolution2D(32, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Convolution2D(16, 3, 3, border_mode='valid'))
    model.add(Activation('relu'))
    model.add(Convolution2D(16, 3, 3))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(32))
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

    model.fit(x0, y0, nb_epoch=nb_epoch, batch_size=128, show_accuracy=True)
    print model.evaluate(x1, y1, batch_size=128)
    return model


def predict(model, fname, is_cnn=False):

    x2 = []

    with open("./test.csv") as f:
        for line in f:
            digit = line.strip().split(",")
            x2.append(digit)
    f.close()
    x2.pop(0)

    x2 = np.array(x2)

    x2 = x2.astype('float32')
    x2 /= 255

    if is_cnn is True:
        x2 = x2.reshape(x2.shape[0], 1, 28, 28)

    r2 = model.predict(x2)

    wf = open("./{}.csv".format(fname), 'w')
    wf.write("ImageId,Label\n")
    imgId = 1
    for r in r2:
        imgLabel = np.argmax(r)
        wf.write(str(imgId)+","+str(imgLabel)+"\n")
        imgId = imgId + 1
    wf.close()


for nb_epoch in range(25, 26):

    adagrad_opt = Adagrad()
    adadelta_opt = Adadelta()
    rmsprop_opt = RMSprop()
    adam_opt = Adam()
    adamax_opt = Adamax()

    # model = cnn(x0, y0, x1, y1, nb_epoch, adagrad_opt)
    # predict(model, "{}_{}".format("cnn_adagrad", nb_epoch), True)

    # model = cnn(x0, y0, x1, y1, nb_epoch, adadelta_opt)
    # predict(model, "{}_{}".format("cnn_adadelta", nb_epoch), True)

    # model = cnn(x0, y0, x1, y1, nb_epoch, rmsprop_opt)
    # predict(model, "{}_{}".format("cnn_rmsprop", nb_epoch), True)

    model = cnn(x0, y0, x1, y1, nb_epoch, adam_opt)
    predict(model, "{}_{}".format("cnn_adam", nb_epoch), True)

    model = cnn(x0, y0, x1, y1, nb_epoch, adamax_opt)
    predict(model, "{}_{}".format("cnn_adamax", nb_epoch), True)
