from __future__ import annotations

from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np

from config import AppConfig
from processing import DataProcessor


class ResultPresenter:
    def __init__(self, config: AppConfig, processor: DataProcessor):
        self.config = config
        self.processor = processor

    def stage_plot(self, results):
        start = self.config.plot_start
        if not results:
            raise ValueError("Нет результатов для построения графиков")

        x_sums = [res[7] for res in results]
        min_time = min(arr.shape[0] for arr in x_sums)
        if min_time <= start:
            raise ValueError(f"Недостаточно временных точек для start={start}: min_time={min_time}")

        t = np.arange(start, min_time, dtype=np.int32)
        trimmed = [arr[start:min_time].astype(np.float32, copy=False) for arr in x_sums]

        x_sum0, x_sum1, x_sum2, x_sum3, x_sum4 = trimmed
        x_dif1 = self.processor.difference(x_sum1, x_sum0)
        x_dif2 = self.processor.difference(x_sum2, x_sum0)
        x_dif3 = self.processor.difference(x_sum3, x_sum0)
        x_dif4 = self.processor.difference(x_sum4, x_sum0)

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

    @staticmethod
    def add_datetime(filename: str, suffix: str) -> str:
        return filename + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "." + suffix

    def save_figures(self, fig1, fig2) -> None:
        self.config.images_path.mkdir(parents=True, exist_ok=True)
        fig1.savefig(self.config.images_path / self.add_datetime("relative", "png"), dpi=self.config.save_dpi)
        fig2.savefig(self.config.images_path / self.add_datetime("absolute", "png"), dpi=self.config.save_dpi)

    @staticmethod
    def show() -> None:
        plt.show()
