import numpy as np
import cv2
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Activation, LeakyReLU
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, BatchNormalization


def neural_network(cnn_type, output, size, filters, kernel_size, num_images, percent_test):

    coeff = int(percent_test * num_images)

    def __init__(self, data_format=None, keepdims=False, **kwargs):
        super(GlobalPooling2D, self).__init__(**kwargs)
        self.data_format = conv_utils.normalize_data_format(data_format)
        self.input_spec = InputSpec(ndim=4)
        self.keepdims = keepdims

    if cnn_type == 1:
        model = Sequential([

            Conv2D(filters, (kernel_size, kernel_size), input_shape=(size, size, 1), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 2, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 3, (kernel_size, kernel_size), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 4, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 5, (kernel_size, kernel_size), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 6, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 7, (kernel_size, kernel_size), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 8, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 8, (kernel_size, kernel_size), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 16, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 16, (kernel_size, kernel_size), activation='relu'),

            GlobalAveragePooling2D(),
            Dense(600, activation='relu'),
            Dropout(0.2),
            Dense(400, activation='relu'),
            Dropout(0.2),
            Dense(500, activation='relu'),
            Dropout(0.2),
            Dense(400, activation='relu'),
            Dropout(0.2),
            Dense(300, activation='relu'),
            Dropout(0.2),
            Dense(200, activation='relu'),
            Dropout(0.2),
            Dense(100, activation='relu'),
            Dense(50, activation='relu'),

        ])

    elif cnn_type == 2:
        model = Sequential([

            Conv2D(filters, (kernel_size, kernel_size), input_shape=(size, size, 1), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 2, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 3, (kernel_size, kernel_size), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 4, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 5, (kernel_size, kernel_size), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 6, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 7, (kernel_size, kernel_size), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 8, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 8, (kernel_size, kernel_size), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 16, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 16, (kernel_size, kernel_size), activation='relu'),

            GlobalAveragePooling2D(),
            Dense(600, activation='relu'),
            Dropout(0.2),
            Dense(400, activation='relu'),
            Dropout(0.2),
            Dense(500, activation='relu'),
            Dropout(0.2),
            Dense(400, activation='relu'),
            Dropout(0.2),
            Dense(300, activation='relu'),
            Dropout(0.2),
            Dense(200, activation='relu'),
            Dropout(0.2),
            Dense(100, activation='relu'),
            Dense(50, activation='relu')

        ])

    elif cnn_type == 3:
        model = Sequential([

            Conv2D(filters, (kernel_size, kernel_size), input_shape=(size, size, 1), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 2, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 3, (kernel_size, kernel_size), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 4, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 5, (kernel_size, kernel_size), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 6, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 7, (kernel_size, kernel_size), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 8, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 8, (kernel_size, kernel_size), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 16, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 16, (kernel_size, kernel_size), activation='relu'),

            GlobalAveragePooling2D(),
            Dense(600, activation='relu'),
            Dropout(0.2),
            Dense(400, activation='relu'),
            Dropout(0.2),
            Dense(500, activation='relu'),
            Dropout(0.2),
            Dense(400, activation='relu'),
            Dropout(0.2),
            Dense(300, activation='relu'),
            Dropout(0.2),
            Dense(200, activation='relu'),
            Dropout(0.2),
            Dense(100, activation='relu'),
            Dense(50, activation='relu')

        ])

    elif cnn_type == 4:
        model = Sequential([

            Conv2D(filters, (kernel_size, kernel_size), input_shape=(size, size, 1), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 2, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 3, (kernel_size, kernel_size), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 4, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 5, (kernel_size, kernel_size), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 6, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 7, (kernel_size, kernel_size), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 8, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 8, (kernel_size, kernel_size), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 16, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 16, (kernel_size, kernel_size), activation='relu'),

            GlobalAveragePooling2D(),
            Dense(600, activation='relu'),
            Dropout(0.2),
            Dense(400, activation='relu'),
            Dropout(0.2),
            Dense(500, activation='relu'),
            Dropout(0.2),
            Dense(400, activation='relu'),
            Dropout(0.2),
            Dense(300, activation='relu'),
            Dropout(0.2),
            Dense(200, activation='relu'),
            Dropout(0.2),
            Dense(100, activation='relu'),
            Dense(50, activation='relu')

        ])

    else:
        model = Sequential([

            Conv2D(filters, (kernel_size, kernel_size), input_shape=(size, size, 1), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 2, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 3, (kernel_size, kernel_size), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 4, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 5, (kernel_size, kernel_size), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 6, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 7, (kernel_size, kernel_size), activation='relu'),
            BatchNormalization(),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 8, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 8, (kernel_size, kernel_size), activation='relu'),
            MaxPooling2D(pool_size=(2, 2)),
            Conv2D(filters * 16, (kernel_size, kernel_size), activation='relu'),
            Conv2D(filters * 16, (kernel_size, kernel_size), activation='relu'),

            GlobalAveragePooling2D(),
            Dense(600, activation='relu'),
            Dropout(0.2),
            Dense(400, activation='relu'),
            Dropout(0.2),
            Dense(500, activation='relu'),
            Dropout(0.2),
            Dense(400, activation='relu'),
            Dropout(0.2),
            Dense(300, activation='relu'),
            Dropout(0.2),
            Dense(200, activation='relu'),
            Dropout(0.2),
            Dense(100, activation='relu'),
            Dense(50, activation='relu')

        ])

    opt = keras.optimizers.RMSprop(lr=0.0001)

    model.add(Dense(units=output.shape[-1]))
    model.compile(optimizer=opt, loss='mse', metrics=['accuracy'])

    return model, coeff

def distance(x1,y1,x2,y2):
    return np.sqrt(np.square(x2-x1) + np.square(y2-y1))