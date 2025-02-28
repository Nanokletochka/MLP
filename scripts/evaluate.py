from train import MLP
from MNIST import *

import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Инициализируем модель
model = MLP(256)

# Инициализируем объект Dataset для обучения
test_dataset = MNIST(is_train=False)
# Инициализируем объект Dataloader для обучения
test_batch_generator = data.DataLoader(test_dataset, 500, shuffle=False, drop_last=False)

# Загружаем веса модели
state_dict = torch.load("../models/best.tar", weights_only=True)
# Загрузить словарь состояний в модель
model.load_state_dict(state_dict)

# Переводим модель в режим тестирования
model.eval()

# Инициализируем списки для хранения истинных и предсказанных значений
y_true_all = []
y_pred_all = []

with torch.no_grad():
    for x_test, y_true in test_batch_generator:
        # Трансформируем
        y_true = y_true.squeeze(-1) # batch_size x 10
        x_test = x_test.float()

        # Предсказываем batch_size x 10
        y_pred_test = model(x_test)

        # Преобразуем предсказания в одномерный список
        y_pred_classes = torch.argmax(y_pred_test, dim=1)
        y_true_classes = torch.argmax(y_true, dim=1)

        # Сохраняем истинные и предсказанные значения
        y_true_all.extend(y_true_classes.cpu().numpy())
        y_pred_all.extend(y_pred_classes.cpu().numpy())

# Рассчитываем метрики
accuracy = accuracy_score(y_true_all, y_pred_all)
precision = precision_score(y_true_all, y_pred_all, average='macro')
recall = recall_score(y_true_all, y_pred_all, average='macro')
f1 = f1_score(y_true_all, y_pred_all, average='macro')
conf_matrix = confusion_matrix(y_true_all, y_pred_all)

# Выводим метрики
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(conf_matrix)

"""Accuracy: 0.9709
Precision: 0.9706
Recall: 0.9706
F1-score: 0.9706
Confusion Matrix:
[[ 969    0    1    0    0    4    2    1    2    1]
 [   1 1121    3    0    1    1    5    2    1    0]
 [   7    3  990    6    4    0    5    5   10    2]
 [   0    1    5  979    0    5    0    6    8    6]
 [   3    0    1    1  959    0    7    2    0    9]
 [   1    1    0   13    4  851    7    2   10    3]
 [   4    3    1    0    4    5  936    0    4    1]
 [   2    3    8    4    6    0    0  990    3   12]
 [   2    0    6    7    5    7    0    2  940    5]
 [   2    1    0    3   15    3    1    3    7  974]]"""