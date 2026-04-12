import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import time
from os import listdir, path
import concurrent.futures

# Настройки отображения графиков (опционально)
plt.rcParams['figure.figsize'] = (14, 6)

start = time.perf_counter()

# --- Константы ---
dm = 1e-3  # Условная погонная масса
p_atm = 101325  # Атмосферное давление [Па]
dt = 1e-7  # Шаг по времени
num_exp_in_folder = 3  # Кол-во экспериментов в папке base (для расчета N_time)
target_y_val = 4.000000000e-02  # Значение для фильтрации строк

# --- Пути к файлам ---
# Используйте r-строки или двойные слеши для Windows путей
directory = r"D:\_work\linses"
base_dir = path.join(directory, "base")

# Списки подпапок
lins_dir = [
    "0_helium_convex",
    "1_helium_concave",
    "2_freon_convex",
    "3_freon_concave",
]
type_exp = 3

# Формирование путей префиксов файлов
# Обратите внимание: os.path.join корректно ставит слеши
prefixes = [
    path.join(base_dir, "NVIS-"),
    path.join(base_dir, "DNVIS-"),
    path.join(base_dir, "DVIS-"),
    path.join(directory, lins_dir[type_exp], "MVIS-"),
    path.join(directory, lins_dir[type_exp], "DMVIS-"),
]

# Определение количества временных шагов
try:
    folder_files = listdir(base_dir)
    # Фильтруем только файлы, если вдруг там есть папки
    file_count = len([f for f in folder_files if path.isfile(path.join(base_dir, f))])
    N_time = int(file_count / num_exp_in_folder)
except FileNotFoundError:
    print(f"Ошибка: Директория {base_dir} не найдена. Установлено N_time = 100 для теста.")
    N_time = 100

print(f"Количество временных шагов: {N_time}")


# --- Функции ---

def get_file_path(prefix, i):
    """Генерирует путь к файлу с правильным добавлением нулей."""
    # i+1, так как нумерация файлов с 1. :04d делает '0001', '0010' и т.д.
    # В оригинале логика была 000+i (до 10), 00+i (до 100). Это эквивалентно 4 знакам.
    suffix = f"{i + 1:04d}"
    # Если в ваших файлах суффикс разной длины (например 001, а не 0001),
    # измените :04d на :03d
    return f"{prefix}{suffix}"


def process_experiment(exp_idx):
    """
    Считывает все файлы для одного эксперимента и считает физику.
    Возвращает массив среднего перемещения x_sum по времени.
    """
    prefix = prefixes[exp_idx]
    print(f"Начало обработки эксперимента {exp_idx} ({prefix})...")

    # Списки для накопления данных
    p_data = []
    y_max_data = []

    for i in range(N_time):
        fpath = get_file_path(prefix, i)
        try:
            # Читаем нужные колонки
            df = pd.read_csv(fpath, header=0, usecols=[1, 2, 3], names=['val', 'y', 'p'], engine='c')

            # --- ИСПРАВЛЕНИЕ: Принудительная конвертация в числа ---
            # errors='coerce' превратит любой "мусор" или текст в NaN (Not a Number)
            df['val'] = pd.to_numeric(df['val'], errors='coerce')
            df['y'] = pd.to_numeric(df['y'], errors='coerce')
            df['p'] = pd.to_numeric(df['p'], errors='coerce')

            # Удаляем строки, где появились NaN (битые данные)
            df = df.dropna()
            # -------------------------------------------------------

            # Фильтрация по значению (теперь это точно числа)
            filtered = df[np.isclose(df['val'], target_y_val)]

            if filtered.empty:
                # Если данных нет, добавляем заглушку, чтобы не ломать массивы
                p_data.append(np.array([p_atm]))
                y_max_data.append(0.0)
            else:
                p_vals = filtered['p'].values
                y_vals = filtered['y'].values * 1e3

                p_data.append(p_vals)
                y_max_data.append(np.max(y_vals))

        except FileNotFoundError:
            print(f"Файл не найден, пропускаем: {fpath}")
            # Добавляем заглушку, чтобы длина массива времени не сбилась
            p_data.append(np.array([p_atm]))
            y_max_data.append(0.0)
            # Если файлов не хватает критически, можно раскомментировать break
            # break

    # --- Расчет физики ---
    if not p_data:
        return np.zeros(N_time)  # Возврат нулей, если данных нет вообще

    x_mean_history = np.zeros(len(p_data))

    # Инициализация массивов скорости и положения (по размеру первого кадра с данными)
    # Находим первый непустой кадр для инициализации размеров
    first_valid_idx = 0
    for idx, p_arr in enumerate(p_data):
        if len(p_arr) > 1 or p_arr[0] != p_atm:
            first_valid_idx = idx
            break

    num_points = len(p_data[first_valid_idx])
    current_v = np.zeros(num_points)
    current_x = np.zeros(num_points)

    for i in range(len(p_data)):
        p_vec = p_data[i]

        # Ресайз массивов, если количество точек изменилось (адаптивная сетка)
        if len(p_vec) != len(current_v):
            # Простая переинициализация под новый размер (с потерей инерции для новых точек)
            # Это компромисс для скорости. В оригинале списки просто росли.
            current_v = np.resize(current_v, len(p_vec))
            current_x = np.resize(current_x, len(p_vec))

        # Теперь p_vec - это массив float, ошибка должна уйти
        a_vec = (p_vec - p_atm) * y_max_data[i] / dm

        if i > 0:
            # Метод трапеций/Эйлера
            new_v = current_v + dt * a_vec
            new_x = current_x + 0.5 * dt * (current_v + new_v)

            current_v = new_v
            current_x = new_x

        x_mean_history[i] = np.mean(current_x)

    print(f"Окончание обработки эксперимента {exp_idx}")
    return x_mean_history


# --- Основной расчет ---

# Рассчитываем 5 экспериментов
results = None
# for k in range(5):
#     print(f"Обработка эксперимента {k} ({prefixes[k]})...")
#     res = process_experiment(k)
#     results.append(res)

if __name__ == '__main__':

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(executor.map(process_experiment, range(5)))

    x_sum0, x_sum1, x_sum2, x_sum3, x_sum4 = results

    # Вектор времени
    t = np.arange(len(x_sum0))

    # Разности (векторное вычитание numpy)
    # В оригинале использовался map(difference...), здесь просто минус
    x_dif1 = x_sum1 - x_sum0
    x_dif2 = x_sum2 - x_sum0
    x_dif3 = x_sum3 - x_sum0
    x_dif4 = x_sum4 - x_sum0

    # --- Построение графиков ---
    print("Построение графиков...")

    # График 1: Относительное перемещение
    fig1, ax1 = plt.subplots()
    ax1.plot(t, x_dif1, color="red", label="Для невязкого с решеткой")
    ax1.plot(t, x_dif2, color="purple", label="Для вязкого с решеткой")
    ax1.plot(t, x_dif3, color="green", label="Для вязкой смеси без решетки")
    ax1.plot(t, x_dif4, color="black", label="Для вязкой смеси с решеткой")

    ax1.set_title('Относительное перемещение "фиктивной" стенки')
    ax1.legend(loc=3)
    ax1.grid()
    ax1.set_xlim(550, N_time)  # Убедитесь, что N_time > 550
    ax1.set_xlabel("t-step")
    ax1.set_ylabel("x, [m]")

    # График 2: Абсолютное перемещение
    fig2, ax2 = plt.subplots()
    ax2.plot(t, x_sum0, color="blue", label="Невязкий без смеси и решетки")
    ax2.plot(t, x_sum1, color="red", label="Невязкий без смеси с решеткой")
    ax2.plot(t, x_sum2, color="purple", label="Вязкий без смеси c решеткой")
    ax2.plot(t, x_sum3, color="green", label="Вязкая смесь без решетки")
    ax2.plot(t, x_sum4, color="black", label="Вязкая смесь с решеткой")

    ax2.set_title('Абсолютное перемещение "фиктивной" стенки')
    ax2.legend(loc=2)
    ax2.grid()
    ax2.set_xlim(550, N_time)
    ax2.set_xlabel("t-step")
    ax2.set_ylabel("x, [m]")

    end = time.perf_counter()
    print(f"Время выполнения (perf_counter): {end - start:.6f} секунд")

    plt.show()

import multiprocessing
import time
import os


def worker_function(message):
    """Функция, которая будет выполняться в отдельном процессе."""
    # Получаем ID текущего процесса
    pid = os.getpid()
    print(f"[{message}] Процесс запущен. PID: {pid}")
    # Имитация выполнения полезной работы
    time.sleep(3)
    print(f"[{message}] Работа завершена.")


if __name__ == '__main__':
    print("Главный процесс запущен. PID:", os.getpid())
    # 1. Создание объектов Process
    # target - функция, которая будет выполнена в новом процессе
    # args - кортеж аргументов для этой функции
    p1 = multiprocessing.Process(target=worker_function, args=('Задача 1',))
    p2 = multiprocessing.Process(target=worker_function, args=('Задача 2',))
    print("Процессы созданы.")

    # 2. Запуск процессов
    p1.start()
    p2.start()
    print("Процессы запущены. Ожидаем завершения...")

    # 3. Ожидание завершения процессов
    # Метод join() блокирует главный процесс до тех пор, пока p1 и p2 не завершат работу.
    p1.join()
    p2.join()
    print("Все дочерние процессы завершили работу.")