from dataclasses import dataclass
from typing import Tuple, Generator, Any

import numpy as np
import pandas as pd
import tensorflow as tf
from numpy import ndarray
from tensorflow.keras import Model

from alexnet import AlexNet
from image_loader import ImageLoader


@dataclass
class ImageAndPrice:
    image: ndarray
    price: float


class VisualFeatureExtractor:
    __prices_dict: dict
    features: ndarray
    mean_squared_error: float

    def __init__(self, images_path, prices_path, number_of_epochs, batch_size, image_size, train_ratio,
                 validation_ratio):
        """
        :param images_path: Path of the images, you can read the whole or part by constraining with a partition
                            e.g. path-to-images/partition=0
        """
        self.__images_path = images_path
        self.__prices_path = prices_path
        self.__number_of_epochs = number_of_epochs
        self.__batch_size = batch_size
        self.__image_size = image_size
        self.__train_ratio = train_ratio
        self.__validation_ratio = validation_ratio
        self.alexnet = AlexNet(image_size + (3,))
        self.__image_loader = ImageLoader(self.__images_path)

    def run(self):
        self.__prices_dict = self.read_prices()

        all_images_gen = self.read_images_as_train_validation_test()
        train_ds, validation_ds, test_ds = (self.build_images_ds(images_gen) for images_gen in all_images_gen)
        no_train_files, no_validation_files, _ = self.__image_loader.get_number_of_files_per_split(
            self.__train_ratio, self.__validation_ratio)

        print(f'{no_train_files} // {self.__batch_size} // {self.__number_of_epochs}')
        print(f'{no_validation_files} // {self.__batch_size} // {self.__number_of_epochs}')

        self.alexnet.model.fit(train_ds, epochs=self.__number_of_epochs, validation_data=validation_ds,
                               steps_per_epoch=no_train_files // (self.__batch_size * self.__number_of_epochs),
                               validation_steps=no_validation_files // (self.__batch_size * self.__number_of_epochs))

        self.mean_squared_error = self.alexnet.model.evaluate(test_ds)

        intermediate_feature_extraction_model = Model(self.alexnet.model.input, self.alexnet.output_layer.output)
        self.features = intermediate_feature_extraction_model.predict(self.build_images_ds(self.read_all_images()))
        np.save('output_features', self.features)

    def read_prices(self) -> dict:
        prices = pd.read_pickle(self.__prices_path, compression='gzip')[['id', 'Price_USD']].set_index('id')
        return dict(zip(prices.index.tolist(), prices['Price_USD'].tolist()))

    def read_images_as_train_validation_test(self):
        all_images_gen = self.__image_loader.load_images_as_train_validation_test(self.__train_ratio,
                                                                                  self.__validation_ratio)
        return (VisualFeatureExtractor.preprocess_images(images, self.__image_size) for images in all_images_gen)

    def read_all_images(self):
        return VisualFeatureExtractor.preprocess_images(self.__image_loader.load_all_images(), self.__image_size)

    @staticmethod
    def preprocess_images(images, size: Tuple[int, int]):
        return (image.copy_with_image(VisualFeatureExtractor.normalize_images(size)(image.image)) for image in images)

    @staticmethod
    def normalize_images(image_size: tuple):
        return lambda image: tf.image.resize(tf.image.per_image_standardization(image), image_size)

    def build_images_ds(self, images_gen):
        return self.create_tf_dataset(self.enrich_images_with_prices(images_gen))

    def enrich_images_with_prices(self, images_gen) -> Generator[ImageAndPrice, Any, None]:
        return (ImageAndPrice(image.image, self.__prices_dict[image.id]) for image in images_gen)

    def create_tf_dataset(self, image_and_prices):
        image_and_price_pairs = ((image_and_price.image, image_and_price.price) for image_and_price in image_and_prices)

        return tf.data.Dataset.from_generator(lambda: image_and_price_pairs, output_signature=(
            tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(), dtype=tf.float64))).batch(batch_size=self.__batch_size)
