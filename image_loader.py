import glob
from dataclasses import dataclass
from os.path import join, sep

import numpy as np
from PIL import Image
from numpy import ndarray


class ImageLoader:
    __path: str

    def __init__(self, path: str):
        self.__path = path

    def load_images_as_train_validation_test(self, train_ratio: float, validation_ratio: float):
        files = self.find_png_files()
        number_of_files = len(files)
        train_part = int(number_of_files * train_ratio)
        validation_part = int(number_of_files * (train_ratio + validation_ratio))
        return (self.__load_images(files[:train_part])), \
               (self.__load_images(files[train_part:validation_part])), \
               (self.__load_images(files[validation_part:]))

    def load_all_images(self):
        return self.__load_images(self.find_png_files())

    @staticmethod
    def __load_images(files):
        return (ImageLoader.__load_image(file_path) for file_path in files)

    def get_number_of_files_per_split(self, train_ratio: float, validation_ratio: float):
        number_of_files = len(self.find_png_files())
        number_of_train_files = int(number_of_files * train_ratio)
        number_of_validation_files = int(number_of_files * (train_ratio + validation_ratio)) - number_of_train_files
        number_of_test_files = number_of_files - number_of_train_files - number_of_validation_files
        return number_of_train_files, number_of_validation_files, number_of_test_files

    def find_png_files(self):
        return glob.glob(join(self.__path, '**', '*.png'), recursive=True)

    @staticmethod
    def __load_image(file_path: str):
        image = Image.open(file_path)
        image = image.convert('RGB') if image.mode != 'RGB' else image
        return ImageAndId(np.asarray(image), int(file_path.split(sep)[-1].split('.')[0]))


@dataclass
class ImageAndId:
    image: ndarray
    id: int

    def copy_with_image(self, image):
        return ImageAndId(image, self.id)
