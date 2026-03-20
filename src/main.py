from __future__ import annotations

import time
from contextlib import contextmanager

from config import AppConfig
from io_module import FileReader
from output import ResultPresenter
from processing import DataProcessor


@contextmanager
def timer(name: str):
    t0 = time.perf_counter()
    yield
    dt = time.perf_counter() - t0
    print(f"{name}: {dt:.6f} s")


class Application:
    def __init__(self):
        self.config = AppConfig().load_from_file()
        self.reader = FileReader(self.config)
        self.processor = DataProcessor(self.config)
        self.presenter = ResultPresenter(self.config, self.processor)

    def run(self) -> None:
        total_start = time.perf_counter()

        with timer("STAGE 1 (READ): read all files"):
            yp = self.reader.stage_read_all()
        print(f"N_time (from data) = {self.config.n_time}")

        with timer("STAGE 2 (COMPUTE): compute from y,p"):
            results = self.processor.stage_compute_all(yp)

        with timer("STAGE 3 (PLOT+SHOW): build plots and show"):
            fig1, fig2 = self.presenter.stage_plot(results)
            self.presenter.save_figures(fig1, fig2)

        total = time.perf_counter() - total_start
        print(f"TOTAL: {total:.6f} s")
        self.presenter.show()


if __name__ == "__main__":
    Application().run()
