# MLP (multilayer perceptron)
Цель проекта обучить многослойный перцептрон (MLP) на наборе данных MNIST с использованием фреймворка PyTorch
и добиться максимального качества предсказания.

## Используемые библиотеки
- torch (2.2.0)
- matplotlib
- tqdm
- sklearn
- numpy

## Структура проекта

- scripts/
    - `MNIST.py`: содержит класс, определяющий интерфейс взаимодействия с данными;
    - `train.py`: содержит класс модели и цикл её обучения;
    - `evaluate.py`: рассчитывает показатели качества (метрики) модели;
    - `predict.py`: запустите данный файл для предсказания цифры на изображении.
- MNIST/
    - Содержит набор рукописных цифр MNIST.
- models/
    - Содержит наилучшую обученную модель (словарь состояний).

## Метрики модели
Accuracy: 0.9709\
Precision: 0.9706\
Recall: 0.9706\
F1-score: 0.9706\
*Были посчитаны на 10 000 тестовых изображениях.*

## Архитектура перцептрона
Наилучший результат показал перцептрон с одним
скрытым слоем на 256 нейронов и скоростью обучения
0.001. Результаты экспериментов с архитектурой перцептрона 
можно найти в каталоге plots\, где название файла 
соответствует числу нейронов в перцептроне.
