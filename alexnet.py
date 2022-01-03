from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Flatten, Dense, BatchNormalization, MaxPool2D, Dropout
from tensorflow.keras.optimizers import Adam


class AlexNet:
    def __init__(self, input_shape):
        super().__init__()
        self.output_layer = Dense(4096, activation='relu')
        self.model = Sequential([
            Conv2D(filters=96, kernel_size=(11, 11), strides=(4, 4), activation='relu', input_shape=input_shape),
            BatchNormalization(),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            Conv2D(filters=256, kernel_size=(5, 5), strides=(1, 1), activation='relu', padding="same"),
            BatchNormalization(),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            BatchNormalization(),
            Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            BatchNormalization(),
            Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            BatchNormalization(),
            MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            self.output_layer,
            Dropout(0.5),
            Dense(1, activation='relu')
        ])

        self.model.summary()
        self.model.compile(optimizer=Adam(), loss='mean_squared_error')
