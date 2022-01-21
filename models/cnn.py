from tensorflow.keras.layers import Activation, Convolution2D, Dropout, Conv2D
from tensorflow.keras.layers import AveragePooling2D, BatchNormalization, Dense
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import SeparableConv2D
from tensorflow.keras import layers
from tensorflow.keras.regularizers import l2
import numpy as np
import matplotlib.pyplot as plt

def cnn(input_shape, num_classes):
    model = Sequential()
    model.add(Conv2D(64, (3, 3), padding="same", input_shape=input_shape, activation='relu'))
    model.add(Conv2D(64, (3, 3), padding="same", activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(Conv2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(32, activation='relu'))
    model.add(Dropout(0.6))
    model.add(Dense(num_classes, activation='softmax'))

    return model


def vgg16(input_shape, num_classes):
    model = Sequential()

    model.add(
        Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", name='block1_conv1',
               input_shape=input_shape))
    model.add(
        Conv2D(64, kernel_size=(3, 3), activation="relu", padding="same", name='block1_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block1_maxpool'))
    model.add(Dropout(.25))

    model.add(
        Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", name='block2_conv1'))
    model.add(
        Conv2D(128, kernel_size=(3, 3), activation="relu", padding="same", name='block2_conv2'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block2_maxpool'))
    model.add(Dropout(.25))

    model.add(
        Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same", name='block3_conv1'))
    model.add(
        Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same", name='block3_conv2'))
    model.add(
        Conv2D(256, kernel_size=(3, 3), activation="relu", padding="same", name='block3_conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block3_maxpool'))
    model.add(Dropout(.25))

    model.add(
        Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name='block4_conv1'))
    model.add(
        Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name='block4_conv2'))
    model.add(
        Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name='block4_conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block4_maxpool'))
    model.add(Dropout(.25))

    model.add(
        Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name='block5_conv1'))
    model.add(
        Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name='block5_conv2'))
    model.add(
        Conv2D(512, kernel_size=(3, 3), activation="relu", padding="same", name='block5_conv3'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='block5_maxpool'))
    model.add(Dropout(.25))

    model.add(Flatten())
    model.add(Dense(4096, activation='relu'))
    model.add(Dense(4096, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))
    return model


def simpler_CNN(input_shape, num_classes):
    model = Sequential()
    model.add(Convolution2D(filters=16, kernel_size=(5, 5), padding='same',
                            name='image_array', input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=16, kernel_size=(5, 5),
                            strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Convolution2D(filters=32, kernel_size=(5, 5), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=32, kernel_size=(5, 5),
                            strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Convolution2D(filters=64, kernel_size=(3, 3), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=64, kernel_size=(3, 3),
                            strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Convolution2D(filters=64, kernel_size=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=128, kernel_size=(3, 3),
                            strides=(2, 2), padding='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(.25))

    model.add(Convolution2D(filters=256, kernel_size=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=128, kernel_size=(3, 3),
                            strides=(2, 2), padding='same'))

    model.add(Convolution2D(filters=256, kernel_size=(1, 1), padding='same'))
    model.add(BatchNormalization())
    model.add(Convolution2D(filters=num_classes, kernel_size=(3, 3),
                            strides=(2, 2), padding='same'))

    model.add(Flatten())
    # model.add(GlobalAveragePooling2D())
    model.add(Activation('softmax', name='predictions'))
    return model

# plots accuracy and loss curves
def plot_model_history(model_history):
    """
    Plot Accuracy and Loss curves given the model_history
    """
    # summarize history for accuracy
    fig, axs = plt.subplots(1, 2, figsize=(15, 5))
    axs[0].plt.plot(model_history.history['accuracy'])
    axs[0].plt.plot(model_history.history['val_accuracy'])
    axs[0].plt.title('model accuracy')
    axs[0].plt.ylabel('accuracy')
    axs[0].plt.xlabel('epoch')
    axs[0].plt.legend(['train', 'test'], loc='upper left')

    # summarize history for loss
    axs[1].plt.plot(model_history.history['loss'])
    axs[1].plt.plot(model_history.history['val_loss'])
    axs[1].plt.title('model accuracy')
    axs[1].plt.ylabel('loss')
    axs[1].plt.xlabel('epoch')
    axs[1].plt.legend(['train', 'test'], loc='upper left')
    fig.savefig('plot.png')
    plt.show()


if __name__ == "__main__":
    input_shape = (48, 48, 1)
    num_classes = 7
    # model = tiny_XCEPTION(input_shape, num_classes)
    # model.summary()
    model = vgg16(input_shape, num_classes)
    model.summary()
    # model = big_XCEPTION(input_shape, num_classes)
    # model.summary()
    # model = simple_CNN((48, 48, 1), num_classes)
    # model.summary()
