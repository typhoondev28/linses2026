import time
from contextlib import contextmanager
from bisect import bisect_left
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


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

num_exp = 3       # кол-во экспериментов в base (как в оригинале)
dt = 1e-7
# N_time = int(2700 / num_exp)

BASE_DIR = Path(__file__).resolve().parent

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
def load_x_index(group_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Читает index.csv из папки группы и возвращает:
      xs        — отсортированный numpy-массив значений x
      filenames — numpy-массив имён parquet-файлов в том же порядке
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

    xs = df["x"].to_numpy(dtype=np.float64, copy=True)
    filenames = df["filename"].to_numpy(dtype=str, copy=True)

    order = np.argsort(xs)
    return xs[order], filenames[order]


def find_filename_for_target_x(group_dir: Path, target_x: float, atol: float = X_INDEX_ATOL) -> Path:
    """
    Находит parquet-файл для target_x через index.csv.
    Используется ближайшее значение с проверкой |x - target_x| <= atol.
    """
    xs, filenames = load_x_index(group_dir)
    if xs.size == 0:
        raise ValueError(f"Индекс пустой: {group_dir / 'index.csv'}")

    xs_list = xs.tolist()
    pos = bisect_left(xs_list, target_x)

    candidate_indices = []
    if pos < xs.size:
        candidate_indices.append(pos)
    if pos > 0:
        candidate_indices.append(pos - 1)

    if not candidate_indices:
        raise ValueError(f"Для {group_dir} не найдено ни одного кандидата для x={target_x:.12e}")

    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    diffs = np.abs(xs[candidate_indices] - target_x)
    best_local = int(np.argmin(diffs))
    best_idx = int(candidate_indices[best_local])
    best_diff = float(diffs[best_local])
    best_x = float(xs[best_idx])
    best_filename = str(filenames[best_idx])

    if best_diff > atol:
        raise ValueError(
            f"В {group_dir / 'index.csv'} не найден x, совпадающий с TARGET_X.\n"
            f"TARGET_X={target_x:.12e}, ближайший x={best_x:.12e}, |dx|={best_diff:.3e}, atol={atol:.3e}"
        )

    parquet_path = group_dir / best_filename
    if not parquet_path.exists():
        raise FileNotFoundError(f"Файл, указанный в индексе, не найден: {parquet_path}")

    return parquet_path


def _reshape_column_by_time(df_sorted: pd.DataFrame, column: str, n_time: int, n_nodes: int, scale: float = 1.0) -> np.ndarray:
    values = df_sorted[column].to_numpy(dtype=np.float64, copy=False)
    if scale != 1.0:
        values = values * scale
    return values.reshape(n_time, n_nodes)


def read_exp_from_indexed_parquet(group_dir: Path, target_x: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Находит нужный parquet по index.csv и читает данные для всех time_id.
    Возвращает уже готовые numpy-массивы:
      y.shape == (n_time, n_nodes)
      p.shape == (n_time, n_nodes)
    """
    parquet_path = find_filename_for_target_x(group_dir, target_x)
    df = pd.read_parquet(parquet_path)

    required_columns = {"time_id", "nodenumber", "y-coordinate", "pressure"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"В {parquet_path} нет нужных колонок {required_columns}. "
            f"Найдены: {list(df.columns)}"
        )

    df = df.sort_values(["time_id", "nodenumber"], kind="stable")

    time_ids = df["time_id"].to_numpy(copy=False)
    unique_time_ids, counts = np.unique(time_ids, return_counts=True)
    if unique_time_ids.size == 0:
        return np.empty((0, 0), dtype=np.float64), np.empty((0, 0), dtype=np.float64)

    if not np.all(counts == counts[0]):
        raise ValueError(
            f"В {parquet_path} разное число узлов для time_id, reshape небезопасен. counts={counts[:10]}"
        )

    n_time = int(unique_time_ids.size)
    n_nodes = int(counts[0])

    y = _reshape_column_by_time(df, "y-coordinate", n_time, n_nodes, scale=1e3)
    p = _reshape_column_by_time(df, "pressure", n_time, n_nodes)
    return y, p


def read_exp(exp: int) -> tuple[np.ndarray, np.ndarray]:
    return read_exp_from_indexed_parquet(MERGED_GROUP_DIRS[exp], TARGET_X)


# -----------------------------
# CPU: вычисления по уже прочитанным y,p
# -----------------------------
def compute_from_yp(y: np.ndarray, p: np.ndarray) -> tuple[
    np.ndarray, np.ndarray, np.ndarray,
    np.ndarray, np.ndarray, np.ndarray
]:
    """
    Полностью повторяет математику исходника values(exp), но НЕ читает файлы.
    Все внутренние вычисления выполняются через numpy arrays.
    Возвращает:
      a, v, x, a_sum, v_sum, x_sum
    """
    y_arr = np.asarray(y, dtype=np.float64)
    p_arr = np.asarray(p, dtype=np.float64)

    if y_arr.size == 0 or p_arr.size == 0:
        empty2d = np.empty((0, 0), dtype=np.float64)
        empty1d = np.empty((0,), dtype=np.float64)
        return empty2d, empty2d, empty2d, empty1d, empty1d, empty1d

    if y_arr.ndim != 2 or p_arr.ndim != 2:
        raise ValueError(f"Ожидались 2D массивы y и p, получено y.ndim={y_arr.ndim}, p.ndim={p_arr.ndim}")
    if y_arr.shape != p_arr.shape:
        raise ValueError(f"Размерности y и p не совпадают: {y_arr.shape} vs {p_arr.shape}")

    n_time, n_nodes = y_arr.shape

    y_max = np.max(y_arr, axis=1)
    a = (p_arr - p_atm) * y_max[:, None] / dm

    v = np.empty_like(a)
    x = np.empty_like(a)
    v[0].fill(0.0)
    x[0].fill(0.0)

    half_dt = 0.5 * dt
    for i in range(1, n_time):
        x_prev = x[i - 1]
        v_prev = v[i - 1]
        a_curr = a[i]

        v_curr = v_prev + dt * a_curr
        x[i] = x_prev + half_dt * v_prev + half_dt * v_curr
        v[i] = v_curr

    a_sum = a.mean(axis=1)
    v_sum = v.mean(axis=1)
    x_sum = x.mean(axis=1)

    return a, v, x, a_sum, v_sum, x_sum


def difference(w: np.ndarray, u: np.ndarray) -> np.ndarray:
    return np.subtract(w, u)


# -----------------------------
# 3 этапа: READ / COMPUTE / PLOT(+SHOW)
# -----------------------------
def stage_read_all() -> list[tuple[np.ndarray, np.ndarray]]:
    return [read_exp(exp) for exp in range(len(MERGED_GROUP_DIRS))]


def stage_compute_all(yp: list[tuple[np.ndarray, np.ndarray]]):
    results = []
    for y, p in yp:
        a, v, x, a_sum, v_sum, x_sum = compute_from_yp(y, p)
        results.append((y, p, a, v, x, a_sum, v_sum, x_sum))
    return results


def stage_plot(results):
    start = 550

    if not results:
        raise ValueError("Нет результатов для построения графиков")

    x_sums = [res[7] for res in results]
    min_time = min(arr.shape[0] for arr in x_sums)
    if min_time <= start:
        raise ValueError(f"Недостаточно временных точек для start={start}: min_time={min_time}")

    t = np.arange(start, min_time, dtype=np.int32)
    trimmed = [arr[start:min_time].astype(np.float32, copy=False) for arr in x_sums]

    x_sum0, x_sum1, x_sum2, x_sum3, x_sum4 = trimmed
    x_dif1 = difference(x_sum1, x_sum0)
    x_dif2 = difference(x_sum2, x_sum0)
    x_dif3 = difference(x_sum3, x_sum0)
    x_dif4 = difference(x_sum4, x_sum0)

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
