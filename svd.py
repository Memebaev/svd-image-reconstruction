import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from io import BytesIO
from tkinter import filedialog, simpledialog, messagebox, Tk
import requests

# Окно для выбора способа загрузки
root = Tk()
root.withdraw()
choice = messagebox.askquestion("Источник изображения", "Загрузить с компьютера?\n(Нет - ввести URL)")

if choice == 'yes':
    path = filedialog.askopenfilename(title="Выберите изображение")
    img = Image.open(path).convert('L')
else:
    url = simpledialog.askstring("Введите URL", "Вставьте ссылку на изображение:")
    response = requests.get(url)
    img = Image.open(BytesIO(response.content)).convert('L')

# Преобразование изображения в массив
A = np.array(img, dtype=np.float64)

# Значения k для SVD
ks = [100, 50, 20, 10, 5] # Кол-во восстанавливаемых сингулярных компонент (если к = None, то покажет оригинал)

# Восстановление изображения с помощью SVD
def svd_re(region, k_val):
    region_centered = region - region.mean()
    U, S, VT = np.linalg.svd(region_centered, full_matrices=False) # SVD
    Ak = U[:, :k_val] @ np.diag(S[:k_val]) @ VT[:k_val, :] # Восстановление по k компонентам
    Ak += region.mean()
    return np.clip(Ak, 0, 255)

# Создание окна из 5 изображений
fig, axs = plt.subplots(1, 5, figsize=(24, 6))
for ax in axs:
    ax.axis('off')

# Увеличение/уменьшение
zoom = 1.0 # Начальный масштаб
min_zoom = 1.0 # Минимальный масштаб
max_zoom = 20.0 # Максимальный масштаб
center_x = A.shape[1] // 2 # Начальная X-координата центра области интереса
center_y = A.shape[0] // 2 # Начальная Y-координата центра области интереса

# Возвращает текущую обрезанную область в соответствии с zoom и центром
def get_cropped():
    h, w = A.shape
    zh = int(h / zoom) # Высота окна просмотра
    zw = int(w / zoom) # Ширина окна просмотра
    y1 = max(0, center_y - zh // 2) # Верхняя граница
    y2 = min(h, center_y + zh // 2) # Нижняя граница
    x1 = max(0, center_x - zw // 2) # Левая граница
    x2 = min(w, center_x + zw // 2) # Правая граница
    return A[y1:y2, x1:x2] # Возврат области

# Обновляет отображение
def update():
    cropped = get_cropped() # Получаем текущее изображение
    for i in range(len(ks)):
        ax = axs[i]
        ax.clear()
        view = svd_re(cropped, ks[i]) # SVD-восстановление
        ax.imshow(view, cmap='gray', vmin=0, vmax=255)
        ax.set_title(f"k={ks[i]}")
        ax.axis('off')
    fig.canvas.draw_idle() # Перерисовка

# Получение координат клика
def get_coords(event):
    if event.inaxes not in axs:
        return None
    idx = axs.tolist().index(event.inaxes)
    if idx >= len(ks):
        return None
    cropped = get_cropped()
    vh, vw = cropped.shape
    offset_x = int(event.xdata / vw * (A.shape[1] / zoom))
    offset_y = int(event.ydata / vh * (A.shape[0] / zoom))
    cx = center_x - (vw // 2) + offset_x # Новая координата центра X
    cy = center_y - (vh // 2) + offset_y # Новая координата центра Y
    return cx, cy

# Обработка кликов мыши для зума
def on_click(event):
    global zoom, center_x, center_y
    coords = get_coords(event)
    if coords is None:
        return
    cx, cy = coords
    if event.button == 1 and zoom < max_zoom: # Левая кнопка мыши - приближение
        zoom *= 1.2
        center_x = int(cx)
        center_y = int(cy)
    elif event.button == 3: # Правая кнопка мыши - отдаление
        zoom /= 1.2
        zoom = max(min_zoom, zoom)
        if zoom == min_zoom: # Возврат к центру изображения
            center_x = A.shape[1] // 2
            center_y = A.shape[0] // 2
    update()

# Клавиши вверх/вниз для изменения k
def on_key(event):
    global ks
    if event.key == 'up': # Увеличить число компонент
        ks = [k + 5 for k in ks]
    elif event.key == 'down': # Уменьшить число компонент
        ks = [max(1, k - 5) for k in ks]
    update()

fig.canvas.mpl_connect('button_press_event', on_click)
fig.canvas.mpl_connect('key_press_event', on_key)

# Отрисовка
update()
plt.tight_layout()
plt.show()