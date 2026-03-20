from __future__ import annotations

from bisect import bisect_left
from pathlib import Path

import numpy as np
import pandas as pd

from config import AppConfig


class FileReader:
    def __init__(self, config: AppConfig):
        self.config = config

    def load_x_index(self, group_dir: Path) -> tuple[np.ndarray, np.ndarray]:
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

    def find_filename_for_target_x(self, group_dir: Path, target_x: float, atol: float) -> Path:
        xs, filenames = self.load_x_index(group_dir)
        if xs.size == 0:
            raise ValueError(f"Индекс пустой: {group_dir / 'index.csv'}")

        pos = bisect_left(xs.tolist(), target_x)
        candidate_indices: list[int] = []
        if pos < xs.size:
            candidate_indices.append(pos)
        if pos > 0:
            candidate_indices.append(pos - 1)
        if not candidate_indices:
            raise ValueError(f"Для {group_dir} не найдено ни одного кандидата для x={target_x:.12e}")

        candidate_indices_arr = np.asarray(candidate_indices, dtype=np.int64)
        diffs = np.abs(xs[candidate_indices_arr] - target_x)
        best_local = int(np.argmin(diffs))
        best_idx = int(candidate_indices_arr[best_local])
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

    @staticmethod
    def _reshape_column_by_time(
        df_sorted: pd.DataFrame,
        column: str,
        n_time: int,
        n_nodes: int,
        scale: float = 1.0,
    ) -> np.ndarray:
        values = df_sorted[column].to_numpy(dtype=np.float64, copy=False)
        if scale != 1.0:
            values = values * scale
        return values.reshape(n_time, n_nodes)

    def read_exp_from_indexed_parquet(self, group_dir: Path, target_x: float) -> tuple[np.ndarray, np.ndarray]:
        parquet_path = self.find_filename_for_target_x(group_dir, target_x, self.config.x_index_atol)
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
            empty = np.empty((0, 0), dtype=np.float64)
            return empty, empty

        if not np.all(counts == counts[0]):
            raise ValueError(
                f"В {parquet_path} разное число узлов для time_id, reshape небезопасен. counts={counts[:10]}"
            )

        n_time = int(unique_time_ids.size)
        n_nodes = int(counts[0])
        y = self._reshape_column_by_time(df, "y-coordinate", n_time, n_nodes, scale=1e3)
        p = self._reshape_column_by_time(df, "pressure", n_time, n_nodes)
        return y, p

    def read_exp(self, exp: int) -> tuple[np.ndarray, np.ndarray]:
        return self.read_exp_from_indexed_parquet(self.config.merged_group_dirs[exp], self.config.target_x)

    def stage_read_all(self) -> list[tuple[np.ndarray, np.ndarray]]:
        yp = [self.read_exp(exp) for exp in range(len(self.config.merged_group_dirs))]
        time_lengths = [int(y.shape[0]) for y, _ in yp if y.ndim == 2]
        if not time_lengths:
            self.config.update_global_n_time(0)
            return yp

        n_time = min(time_lengths)
        self.config.update_global_n_time(n_time)

        unique_lengths = sorted(set(time_lengths))
        if len(unique_lengths) > 1:
            print(
                "WARNING: в файлах разное число временных шагов: "
                f"{unique_lengths}. N_time установлен в минимальное значение {n_time}."
            )
        return yp
