from MNIST import *

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

from tqdm import tqdm

import matplotlib.pyplot as plt


class MLP(nn.Module):
    def __init__(self, l1_neurons):
        super().__init__()

        # Кол-во нейронов скрытого слоя
        self.l1_neurons = l1_neurons

        # Скрытые слои
        self.layer_1 = nn.Linear(28 * 28, self.l1_neurons)
        # Выходной слой
        self.output_layer = nn.Linear(self.l1_neurons, 10)

    def forward(self, x):
        # Выход первого слоя
        z1 = self.layer_1.forward(x)
        u1 = F.relu(z1)
        # Выход последнего слоя
        z2 = self.output_layer.forward(u1)

        return z2


# Обучаем модель: 1 слой на 32 нейрона
# Epoch 5/5: 100%|██████████| 120/120 [02:12<00:00,  1.10s/batch, Loss=1.69]
# Epoch 15/15: 100%|██████████| 120/120 [02:06<00:00,  1.05s/batch, Train Loss=0.618]
model = MLP(32)
model.train()

# Инициализируем объект Dataset для обучения
train_dataset = MNIST()

# Инициализируем объект Dataloader для обучения
train_batch_generator = data.DataLoader(
    train_dataset,
    500,
    shuffle=True,
    drop_last=False
)

# Инициализируем объект Dataset для обучения
test_dataset = MNIST(is_train=False)

# Инициализируем объект Dataloader для обучения
test_batch_generator = data.DataLoader(
    test_dataset,
    500,
    shuffle=True,
    drop_last=False
)

# Запись истории
loss_train_history = []
loss_test_history = []

# Оптимизатор
optim = optim.Adam(model.parameters(), lr=0.01)

# Выбираем функцию потерь
loss = nn.CrossEntropyLoss()

# Кол-во эпох обучения
epochs = 8

# Градиентный спуск
for epoch in range(epochs):
    # Обернем batch_generator в tqdm для отображения прогресса
    train_loss_epoch = 0  # Для накопления ошибки на трейне за эпоху
    with tqdm(train_batch_generator, desc=f"Epoch {epoch + 1}/{epochs}", unit="batch") as pbar:
        for x_train, y_train in pbar:
            # Преобразуем массив y_train
            y_train = y_train.squeeze(-1)
            # Преобразуем в нужный тип
            x_train = x_train.float()
            # Получаем предсказания модели
            y_pred = model(x_train)
            # Функция потерь
            loss_value = loss(y_pred, y_train)
            # Обнуляем градиенты
            optim.zero_grad()
            # Вычисляем градиент по весам
            loss_value.backward()
            # Обновляем веса
            optim.step()

            # Накопление ошибки на трейне за эпоху
            train_loss_epoch += loss_value.item()

            # Обновляем описание прогресс-бара с текущим значением потерь
            pbar.set_postfix({"Train Loss": loss_value.item()})

    # Средняя ошибка на трейне за эпоху
    train_loss_epoch /= len(train_batch_generator)
    loss_train_history.append(train_loss_epoch)  # Записываем ошибку на трейне

    # После каждой эпохи вычисляем ошибку на тестовом наборе
    model.eval()  # Переводим модель в режим оценки
    test_loss = 0
    with torch.no_grad():  # Отключаем вычисление градиентов
        for x_test, y_test in test_batch_generator:
            y_test = y_test.squeeze(-1)
            x_test = x_test.float()
            y_pred_test = model(x_test)
            test_loss += loss(y_pred_test, y_test).item()

    # Средняя ошибка на тестовом наборе
    test_loss /= len(test_batch_generator)
    loss_test_history.append(test_loss)  # Записываем ошибку на тесте
    model.train()  # Возвращаем модель в режим обучения


# Построение графиков истории ошибок
plt.figure(figsize=(10, 6))  # Задаём размер графика

# График ошибки на трейне
plt.plot(loss_train_history, label="Train Loss", marker="o", linestyle="-", color="blue")

# График ошибки на тесте
plt.plot(loss_test_history, label="Test Loss", marker="o", linestyle="-", color="red")

# Добавляем подписи
plt.title("Train and Test Loss over Epochs")  # Заголовок
plt.xlabel("Epoch")  # Ось X: эпохи
plt.ylabel("Loss")  # Ось Y: значение ошибки
plt.legend()  # Легенда
plt.grid(True)  # Сетка для удобства

# Показываем график
plt.show()
