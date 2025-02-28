import torch.utils.data as data

import os
from PIL import Image
import json
import numpy as np


class MNIST(data.Dataset):
    """Класс MNIST определяет интерфейс взаимодействия с набором данных на внешнем носителе."""

    def __init__(self, is_train=True):
        # Путь до обучающего и тестового набора данных
        self.path_to_data = os.path.join("../MNIST", "dataset")

        # Определяем целевую папку
        self.target_folder = "train" if is_train else "test"

        # Словарь, содержащий кол-во изображений
        # каждого класса: {<class_folder_name>:<number_of_images>}
        self.class_len = {}

        # Проходимся по папке каждого класса и считаем кол-во изображений в ней
        for class_folder in os.listdir(os.path.join(self.path_to_data, self.target_folder)):
            path_to_class_folder = os.path.join(self.path_to_data, self.target_folder, class_folder)
            self.class_len[class_folder] = len(os.listdir(path_to_class_folder))

    def __getitem__(self, item):
        # Накапливающаяся сумма изображений по классам
        elem_sum = 0

        # Проходимся по всем классам словаря class_len
        for class_name, num_of_images in self.class_len.items():
            elem_sum += num_of_images

            # Находим папку классу, из которой будем брать изображение
            if item < elem_sum:
                # Порядковый номер изображения для данной папки класса
                elem_index = self.class_len[class_name] - (elem_sum - item)

                # Список всех изображений папки класса
                path_to_class_folder = os.path.join(self.path_to_data, self.target_folder, class_name)

                # Список всех имён изображений данной папки
                list_of_images = os.listdir(path_to_class_folder)

                # Открываем изображение
                image_path = os.path.join(self.path_to_data, self.target_folder, class_name, list_of_images[elem_index])
                image = Image.open(image_path)

                return self.image_to_vector(image), self.create_target_vector(class_name)

        # Вызываем ошибку, если индекс выходит за возможный диапазон
        raise ValueError('Index out of list range.')

    def __len__(self):
        return sum(self.class_len.values())

    def create_target_vector(self, class_folder_name):
        """
        Возвращает numpy вектор 10x1 целевой переменной.

        Например, для класса с меткой "2" вернётся:
        0 0 1 0 0 0 0 0 0 0
        """

        # Создаём вектор из нулей
        zero = np.zeros((10, 1))

        # На соответствующей данному классу позиции заменяем 0 на 1
        with open(os.path.join(self.path_to_data, "format.json"), 'r', encoding='utf-8') as file:
            format_data = json.load(file)
            target_index = format_data[class_folder_name]
            zero[target_index, 0] = 1

            return zero

    @staticmethod
    def image_to_vector(image):
        """Разворачивает изображение 28х28 в одномерный numpy вектор."""

        # Преобразуем изображение в массив NumPy
        img_array = np.array(image)

        # Разворачиваем массив в вектор
        img_vector = img_array.flatten()

        return img_vector

