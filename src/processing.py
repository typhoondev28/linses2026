from __future__ import annotations

import numpy as np

from config import AppConfig


class DataProcessor:
    def __init__(self, config: AppConfig):
        self.config = config

    def compute_from_yp(self, y: np.ndarray, p: np.ndarray):
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

        n_time, _ = y_arr.shape
        y_max = np.max(y_arr, axis=1)
        a = (p_arr - self.config.p_atm) * y_max[:, None] / self.config.dm

        v = np.empty_like(a)
        x = np.empty_like(a)
        v[0].fill(0.0)
        x[0].fill(0.0)

        half_dt = 0.5 * self.config.dt
        for i in range(1, n_time):
            x_prev = x[i - 1]
            v_prev = v[i - 1]
            a_curr = a[i]

            v_curr = v_prev + self.config.dt * a_curr
            x[i] = x_prev + half_dt * v_prev + half_dt * v_curr
            v[i] = v_curr

        a_sum = a.mean(axis=1)
        v_sum = v.mean(axis=1)
        x_sum = x.mean(axis=1)
        return a, v, x, a_sum, v_sum, x_sum

    @staticmethod
    def difference(w: np.ndarray, u: np.ndarray) -> np.ndarray:
        return np.subtract(w, u)

    def stage_compute_all(self, yp: list[tuple[np.ndarray, np.ndarray]]):
        results = []
        for y, p in yp:
            a, v, x, a_sum, v_sum, x_sum = self.compute_from_yp(y, p)
            results.append((y, p, a, v, x, a_sum, v_sum, x_sum))
        return results
