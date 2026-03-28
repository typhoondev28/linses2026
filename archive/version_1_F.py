import csv
import time
from contextlib import contextmanager
from os import listdir
import pandas as pd
import numpy as np
from datetime import datetime
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

ad = ["000", "00", "0"]

# Получаем список файлов и вычисляем, сколько "временных" файлов в одном эксперименте
# folder = listdir(base)
num_exp = 3  # кол-во экспериментов в base (как в оригинале)
dt = 1e-7
N_time = int(2700 / num_exp)


from pathlib import Path

# Папка со скриптом → берём от неё путь к linses
BASE_DIR = Path(__file__).resolve().parent

merged_path = [
    BASE_DIR / "linses" / "merged_parquet" / "base" / "NVIS.parquet",
    BASE_DIR / "linses" / "merged_parquet" / "base" / "DNVIS.parquet",
    BASE_DIR / "linses" / "merged_parquet" / "base" / "DVIS.parquet",
    BASE_DIR / "linses" / "merged_parquet" / "3_freon_concave" / "MVIS.parquet",
    BASE_DIR / "linses" / "merged_parquet" / "3_freon_concave" / "DMVIS.parquet",
]

IMAGES_PATH = BASE_DIR / "images"

# -----------------------------
# I/O: чтение файлов
# -----------------------------
def file_name(i: int, path_prefix: str, ext: str = "") -> str:
    i += 1
    if i < 10:
        add = "000" + str(i)
    elif i < 100:
        add = "00" + str(i)
    elif i < 1000:
        add = "0" + str(i)
    else:
        add = str(i)
    return path_prefix + add + ext


TARGET_X = 4.0e-2


def read_exp_from_big_parquet(parquet_path: str):
    df = pd.read_parquet(parquet_path)

    y = []
    p = []

    for _, group in df.groupby("time_id", sort=True):
        group = group.sort_values("nodenumber")
        y.append((group["y-coordinate"].to_numpy() * 1e3).tolist())
        p.append(group["pressure"].to_numpy().tolist())

    return y, p

def read_one_timefile_parquet(i: int, path_prefix: str) -> tuple[list[float], list[float]]:
    fname = file_name(i, path_prefix, ext=".parquet")

    # читаем только нужные колонки (быстро и экономно)
    df = pd.read_parquet(fname, columns=["x-coordinate", "y-coordinate", "pressure"])

    x = df["x-coordinate"].to_numpy()
    m = np.isclose(x, TARGET_X, rtol=0.0, atol=1e-9)

    y = (df["y-coordinate"].to_numpy()[m] * 1e3)
    p = df["pressure"].to_numpy()[m]

    return y.tolist(), p.tolist()

def read_one_timefile_pd(i: int, path_prefix: str) -> tuple[list[float], list[float]]:
    fname = file_name(i, path_prefix)

    df = pd.read_csv(
        fname,
        header=0,
        usecols=[1, 2, 3],          # x, y, pressure по индексам
        skipinitialspace=True,
        dtype="float32",
        engine="c",
        memory_map = True
    )

    x = df.iloc[:, 0].to_numpy()
    y = df.iloc[:, 1].to_numpy()
    p = df.iloc[:, 2].to_numpy()

    m = np.isclose(x, 4.0e-2, rtol=0.0, atol=1e-9)

    return (y[m] * 1e3).tolist(), p[m].tolist()

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


def read_exp(exp: int):
    return read_exp_from_big_parquet(merged_path[exp])


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


def stage_plot(results):
    start = 550

    t = np.arange(N_time, dtype=np.int32)[start:]

    x_sum0 = np.asarray(results[0][7], dtype=np.float32)[start:]
    x_sum1 = np.asarray(results[1][7], dtype=np.float32)[start:]
    x_sum2 = np.asarray(results[2][7], dtype=np.float32)[start:]
    x_sum3 = np.asarray(results[3][7], dtype=np.float32)[start:]
    x_sum4 = np.asarray(results[4][7], dtype=np.float32)[start:]

    x_dif1 = x_sum1 - x_sum0
    x_dif2 = x_sum2 - x_sum0
    x_dif3 = x_sum3 - x_sum0
    x_dif4 = x_sum4 - x_sum0

    fig1, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(t, x_dif1, label="Для невязкого с решеткой")
    ax1.plot(t, x_dif2, label="Для вязкого с решеткой")
    ax1.plot(t, x_dif3, label="Для вязкой смеси без решетки")
    ax1.plot(t, x_dif4, label="Для вязкой смеси с решеткой")
    ax1.set_title('Относительное перемещение "фиктивной" стенки')
    ax1.set_xlabel("t-step")
    ax1.set_ylabel("x, [m]")
    ax1.legend(loc=3)
    ax1.grid()

    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.plot(t, x_sum0, label="Невязкий без смеси и решетки")
    ax2.plot(t, x_sum1, label="Невязкий без смеси с решеткой")
    ax2.plot(t, x_sum2, label="Вязкий без смеси c решеткой")
    ax2.plot(t, x_sum3, label="Вязкая смесь без решетки")
    ax2.plot(t, x_sum4, label="Вязкая смесь с решеткой")
    ax2.set_title('Абсолютное перемещение "фиктивной" стенки')
    ax2.set_xlabel("t-step")
    ax2.set_ylabel("x, [m]")
    ax2.legend(loc=2)
    ax2.grid()

    return fig1, fig2


def stage_show():
    plt.show()

def add_datetime(filename, suffix):
    return filename + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + suffix

def main():
    total_start = time.perf_counter()

    with timer("STAGE 1 (READ): read all files"):
        yp = stage_read_all()

    with timer("STAGE 2 (COMPUTE): compute from y,p"):
        results = stage_compute_all(yp)

    with timer("STAGE 3 (PLOT+SHOW): build plots and show"):
        fig1, fig2 = stage_plot(results)
        fig1.savefig(IMAGES_PATH / add_datetime("relative", "png"), dpi=120)
        fig2.savefig(IMAGES_PATH / add_datetime("absolute", "png"), dpi=120)
        # plt.close("all")

    total = time.perf_counter() - total_start
    print(f"TOTAL: {total:.6f} s")


    stage_show()

if __name__ == "__main__":
    main()