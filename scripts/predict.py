from train import MLP

from PIL import Image
import numpy as np
import torch

# Определяем путь до изобжраения
file_path = "Путь_до_файла"

# Открываем изображение
img = Image.open(file_path)
# Преобразуем в градацию серого
img = img.convert("L")
# Разворачиваем в вектор
img_array = np.array(img).flatten()
# Превращаем в тензор
img_vector = torch.Tensor(img_array)

# Прогоняем через модель
model = MLP(256)
# Загружаем веса модели
state_dict = torch.load("../models/best.tar", weights_only=True)
# Загрузить словарь состояний в модель
model.load_state_dict(state_dict)
# Переводим модель в режим тестирования
model.eval()

# Предсказываем
with torch.no_grad():
    y_pred = model(img_vector)
    print(torch.argmax(y_pred).item())
