from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense



class CNN:
    @staticmethod
    def build(width, height, depth, total_classes, save_weights_path=None):
        model = Sequential()

        # Conv2D signature: Conv2D(filters, kernel_size, ...)
        model.add(Conv2D(20, (5, 5), padding="same",
                         input_shape=(height, width, depth)))  # (28,28,1)
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(50, (5, 5), padding="same"))
        model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Conv2D(100, (5, 5), padding="same"))
        model.add(Activation("relu"))
        # *** height & width are 7×7 here, so 2 × 2 pooling is still safe ***
        model.add(MaxPooling2D(pool_size=(2, 2)))

        model.add(Flatten())
        model.add(Dense(500, activation="relu"))
        model.add(Dense(total_classes, activation="softmax"))

        if save_weights_path is not None:
            model.load_weights(save_weights_path)
        return model
