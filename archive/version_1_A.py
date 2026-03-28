import csv
import time
from contextlib import contextmanager
from os import listdir

import matplotlib.pyplot as plt


@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"{name}: {dt:.6f} s")


# -----------------------------
# Константы и настройки
# -----------------------------
dm = 1e-3         # условная погонная масса
p_atm = 101325    # атмосферное давление [Па]

directory = "D:\\_work\\linses\\linses_base\\"
base = "D:\\_work\\linses\\linses_base\\base\\"

lins_dir = [
    "0_helium_convex\\",
    "1_helium_concave\\",
    "2_freon_convex\\",
    "3_freon_concave\\",
]
type_exp = 3

path = [
    directory + "base\\" + "NVIS-",
    directory + "base\\" + "DNVIS-",
    directory + "base\\" + "DVIS-",
    directory + lins_dir[type_exp] + "MVIS-",
    directory + lins_dir[type_exp] + "DMVIS-",
]

ad = ["000", "00", "0"]

# Получаем список файлов и вычисляем, сколько "временных" файлов в одном эксперименте
folder = listdir(base)
num_exp = 3  # кол-во экспериментов в base (как в оригинале)
dt = 1e-7
N_time = int(len(folder) / num_exp) - 1
print(N_time)


# -----------------------------
# I/O: чтение файлов
# -----------------------------
def file_name(i: int, path_prefix: str) -> str:
    """Как в оригинале: NVIS-001, NVIS-010, ..."""
    i += 1
    if i < 10:
        add = ad[0] + str(i)
    elif i < 100:
        add = ad[1] + str(i)
    elif i < 1000:
        add = ad[2] + str(i)
    else:
        add = str(i)
    return path_prefix + add


def read_one_timefile(i: int, path_prefix: str) -> tuple[list[float], list[float]]:
    """
    Читает один временной файл и возвращает (y_i, p_i).
    Фильтр по lines[1] == 0.04 как в исходнике.
    """
    y_i: list[float] = []
    p_i: list[float] = []

    with open(file_name(i, path_prefix), mode="r") as f:
        data = csv.reader(f)
        next(data)  # пропускаем заголовок

        for lines in data:
            if float(lines[1]) == 4.000000000e-02:
                p_i.append(float(lines[3]))
                y_i.append(float(lines[2]) * 1e3)

    return y_i, p_i


def read_exp(exp: int) -> tuple[list[list[float]], list[list[float]]]:
    """
    Читает ВСЕ временные файлы для одного exp и возвращает массивы:
      y[time][point], p[time][point]
    """
    y: list[list[float]] = []
    p: list[list[float]] = []

    prefix = path[exp]
    points_count = 0
    for i in range(N_time):
        y_i, p_i = read_one_timefile(i, prefix)
        y.append(y_i)
        p.append(p_i)
        points_count += len(y_i)

    print(f"exp={exp} points: {points_count}")
    return y, p


# -----------------------------
# CPU: вычисления по уже прочитанным y,p
# -----------------------------
def compute_from_yp(y: list[list[float]], p: list[list[float]]) -> tuple[
    list[list[float]], list[list[float]], list[list[float]],
    list[float], list[float], list[float]
]:
    """
    Полностью повторяет математику исходника values(exp), но НЕ читает файлы.
    Возвращает:
      a, v, x, a_sum, v_sum, x_sum
    """
    a: list[list[float]] = []
    v: list[list[float]] = []
    x: list[list[float]] = []

    a_sum = [0.0] * N_time
    v_sum = [0.0] * N_time
    x_sum = [0.0] * N_time

    # В оригинале среднее делится на len(y[0]) для всех i
    denom = len(y[0]) if y and y[0] else 1

    for i in range(N_time):
        a.append([])
        v.append([])
        x.append([])

        # max(y[i]) в оригинале вычисляется в каждой итерации j,
        # но это константа для данного i — считаем один раз.
        y_max = max(y[i]) if y[i] else 0.0

        for j in range(len(y[i])):
            if i == 0:
                x[i].append(0.0)
                v[i].append(0.0)
            else:
                x[i].append(x[i - 1][j] + 0.5 * dt * v[i - 1][j])

            # ускорение
            a_ij = (p[i][j] - p_atm) * y_max / dm
            a[i].append(a_ij)

            # скорость
            if i > 0:
                v_ij = v[i - 1][j] + dt * a_ij
                v[i].append(v_ij)
            else:
                v_ij = v[i][j]

            # перемещение (второй полушаг)
            x[i][j] += 0.5 * dt * v_ij

            # средние
            a_sum[i] += a_ij / denom
            v_sum[i] += v_ij / denom
            x_sum[i] += x[i][j] / denom

    return a, v, x, a_sum, v_sum, x_sum


def difference(w, u):
    return w - u


# -----------------------------
# 3 этапа: READ / COMPUTE / PLOT(+SHOW)
# -----------------------------
def stage_read_all():
    yp = []
    for exp in range(5):
        yp.append(read_exp(exp))
    return yp


def stage_compute_all(yp):
    results = []
    for exp in range(5):
        y, p = yp[exp]
        a, v, x, a_sum, v_sum, x_sum = compute_from_yp(y, p)
        results.append((y, p, a, v, x, a_sum, v_sum, x_sum))
    return results


def stage_plot_and_show(results):
    t = list(range(N_time))

    x_sum0 = results[0][7]
    x_sum1 = results[1][7]
    x_sum2 = results[2][7]
    x_sum3 = results[3][7]
    x_sum4 = results[4][7]

    x_dif1 = list(map(difference, x_sum1, x_sum0))
    x_dif2 = list(map(difference, x_sum2, x_sum0))
    x_dif3 = list(map(difference, x_sum3, x_sum0))
    x_dif4 = list(map(difference, x_sum4, x_sum0))

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t, x_dif1, color="red")
    ax.plot(t, x_dif2, color="purple")
    ax.plot(t, x_dif3, color="green")
    ax.plot(t, x_dif4, color="black")

    ax.set_title('Относительное перемещение "фиктивной" стенки')
    ax.legend(
        [
            "Для невязкого с решеткой",
            "Для вязкого с решеткой",
            "Для вязкой смеси без решетки",
            "Для вязкой смеси с решеткой",
        ],
        loc=3,
    )
    ax.grid()
    ax.set_xlim(550, N_time)
    ax.set_xlabel("t-step")
    ax.set_ylabel("x, [m]")

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(t, x_sum0, color="blue")
    ax.plot(t, x_sum1, color="red")
    ax.plot(t, x_sum2, color="purple")
    ax.plot(t, x_sum3, color="green")
    ax.plot(t, x_sum4, color="black")

    ax.set_title('Абсолютное перемещение "фиктивной" стенки')
    ax.legend(
        [
            "Невязкий без смеси и решетки",
            "Невязкий без смеси с решеткой",
            "Вязкий без смеси c решеткой",
            "Вязкая смесь без решетки",
            "Вязкая смесь с решеткой",
        ],
        loc=2,
    )
    ax.grid()
    ax.set_xlim(550, N_time)
    ax.set_xlabel("t-step")
    ax.set_ylabel("x, [m]")

    # plt.show()


def main():
    total_start = time.perf_counter()

    with timer("STAGE 1 (READ): read all files"):
        yp = stage_read_all()

    with timer("STAGE 2 (COMPUTE): compute from y,p"):
        results = stage_compute_all(yp)

    with timer("STAGE 3 (PLOT+SHOW): build plots and show"):
        stage_plot_and_show(results)

    total = time.perf_counter() - total_start
    print(f"TOTAL: {total:.6f} s")

    plt.show()

if __name__ == "__main__":
    main()