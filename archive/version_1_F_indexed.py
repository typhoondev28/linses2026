import csv
import time
from contextlib import contextmanager
from bisect import bisect_left
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from pathlib import Path


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

# Получаем список файлов и вычисляем, сколько "временных" файлов в одном эксперименте
num_exp = 3  # кол-во экспериментов в base (как в оригинале)
dt = 1e-7
N_time = int(2700 / num_exp)

# Папка со скриптом → берём от неё путь к linses
BASE_DIR = Path(__file__).resolve().parent

# Новая структура данных: в каждой папке лежат index.csv и parquet-файлы по отдельным X
MERGED_GROUP_DIRS = [
    BASE_DIR / "linses" / "merged_parquet_full" / "base" / "NVIS",
    BASE_DIR / "linses" / "merged_parquet_full" / "base" / "DNVIS",
    BASE_DIR / "linses" / "merged_parquet_full" / "base" / "DVIS",
    BASE_DIR / "linses" / "merged_parquet_full" / "3_freon_concave" / "MVIS",
    BASE_DIR / "linses" / "merged_parquet_full" / "3_freon_concave" / "DMVIS",
]

IMAGES_PATH = BASE_DIR / "images"

TARGET_X = 4.0e-2
X_INDEX_ATOL = 1e-12


# -----------------------------
# I/O: поиск parquet по index.csv
# -----------------------------
def load_x_index(group_dir: Path) -> tuple[list[float], list[str]]:
    """
    Читает index.csv из папки группы и возвращает:
      xs        — отсортированный список значений x
      filenames — имена parquet-файлов в том же порядке
    """
    index_path = group_dir / "index.csv"
    if not index_path.exists():
        raise FileNotFoundError(f"Не найден индекс-файл: {index_path}")

    df = pd.read_csv(index_path)

    required_columns = {"x", "filename"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"В {index_path} нет нужных колонок {required_columns}. "
            f"Найдены: {list(df.columns)}"
        )

    xs = df["x"].astype(float).tolist()
    filenames = df["filename"].astype(str).tolist()

    pairs = sorted(zip(xs, filenames), key=lambda t: t[0])
    xs_sorted = [p[0] for p in pairs]
    files_sorted = [p[1] for p in pairs]
    return xs_sorted, files_sorted


def find_filename_for_target_x(group_dir: Path, target_x: float, atol: float = X_INDEX_ATOL) -> Path:
    """
    Находит parquet-файл для target_x через index.csv.
    Используется ближайшее значение с проверкой |x - target_x| <= atol.
    """
    xs, filenames = load_x_index(group_dir)
    if not xs:
        raise ValueError(f"Индекс пустой: {group_dir / 'index.csv'}")

    pos = bisect_left(xs, target_x)

    candidates = []
    if pos < len(xs):
        candidates.append((abs(xs[pos] - target_x), xs[pos], filenames[pos]))
    if pos > 0:
        candidates.append((abs(xs[pos - 1] - target_x), xs[pos - 1], filenames[pos - 1]))

    if not candidates:
        raise ValueError(f"Для {group_dir} не найдено ни одного кандидата для x={target_x:.12e}")

    best_diff, best_x, best_filename = min(candidates, key=lambda t: t[0])

    if best_diff > atol:
        raise ValueError(
            f"В {group_dir / 'index.csv'} не найден x, совпадающий с TARGET_X.\n"
            f"TARGET_X={target_x:.12e}, ближайший x={best_x:.12e}, |dx|={best_diff:.3e}, atol={atol:.3e}"
        )

    parquet_path = group_dir / best_filename
    if not parquet_path.exists():
        raise FileNotFoundError(
            f"Файл, указанный в индексе, не найден: {parquet_path}"
        )

    return parquet_path


def read_exp_from_indexed_parquet(group_dir: Path, target_x: float):
    """
    Находит нужный parquet по index.csv и читает данные для всех time_id.
    В parquet уже лежат только данные для одного X, поэтому дополнительная фильтрация не нужна.
    """
    parquet_path = find_filename_for_target_x(group_dir, target_x)
    df = pd.read_parquet(parquet_path)

    y = []
    p = []

    for _, group in df.groupby("time_id", sort=True):
        group = group.sort_values("nodenumber")
        y.append((group["y-coordinate"].to_numpy() * 1e3).tolist())
        p.append(group["pressure"].to_numpy().tolist())

    return y, p


def read_exp(exp: int):
    return read_exp_from_indexed_parquet(MERGED_GROUP_DIRS[exp], TARGET_X)


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
    return filename + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "." + suffix


def main():
    total_start = time.perf_counter()

    with timer("STAGE 1 (READ): read all files"):
        yp = stage_read_all()

    with timer("STAGE 2 (COMPUTE): compute from y,p"):
        results = stage_compute_all(yp)

    with timer("STAGE 3 (PLOT+SHOW): build plots and show"):
        fig1, fig2 = stage_plot(results)
        IMAGES_PATH.mkdir(parents=True, exist_ok=True)
        fig1.savefig(IMAGES_PATH / add_datetime("relative", "png"), dpi=120)
        fig2.savefig(IMAGES_PATH / add_datetime("absolute", "png"), dpi=120)

    total = time.perf_counter() - total_start
    print(f"TOTAL: {total:.6f} s")

    stage_show()


if __name__ == "__main__":
    main()
