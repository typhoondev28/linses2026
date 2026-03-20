from __future__ import annotations

from configparser import ConfigParser
from pathlib import Path
from typing import List


# -----------------------------
# Глобальные переменные настроек
# -----------------------------
dm = 1e-3
p_atm = 101325.0
num_exp = 3
dt = 1e-7
N_time = None

BASE_DIR = Path(__file__).resolve().parent.parent
MERGED_GROUP_DIRS: List[Path] = [
    BASE_DIR / "linses" / "merged_parquet_full" / "base" / "NVIS",
    BASE_DIR / "linses" / "merged_parquet_full" / "base" / "DNVIS",
    BASE_DIR / "linses" / "merged_parquet_full" / "base" / "DVIS",
    BASE_DIR / "linses" / "merged_parquet_full" / "3_freon_concave" / "MVIS",
    BASE_DIR / "linses" / "merged_parquet_full" / "3_freon_concave" / "DMVIS",
]
IMAGES_PATH = BASE_DIR / "images"
TARGET_X = 4.0e-2
X_INDEX_ATOL = 1e-12
PLOT_START = 550
SAVE_DPI = 120


class AppConfig:
    def __init__(self, config_path: Path | None = None):
        self.base_dir = BASE_DIR
        self.config_path = config_path or (self.base_dir / "config.ini")

        self.dm = dm
        self.p_atm = p_atm
        self.num_exp = num_exp
        self.dt = dt
        self.n_time = N_time
        self.merged_group_dirs = list(MERGED_GROUP_DIRS)
        self.images_path = IMAGES_PATH
        self.target_x = TARGET_X
        self.x_index_atol = X_INDEX_ATOL
        self.plot_start = PLOT_START
        self.save_dpi = SAVE_DPI

    def load_from_file(self) -> "AppConfig":
        parser = ConfigParser()
        if not self.config_path.exists():
            return self

        parser.read(self.config_path, encoding="utf-8")

        if parser.has_section("physics"):
            self.dm = parser.getfloat("physics", "dm", fallback=self.dm)
            self.p_atm = parser.getfloat("physics", "p_atm", fallback=self.p_atm)
            self.dt = parser.getfloat("physics", "dt", fallback=self.dt)

        if parser.has_section("input"):
            self.target_x = parser.getfloat("input", "target_x", fallback=self.target_x)
            self.x_index_atol = parser.getfloat("input", "x_index_atol", fallback=self.x_index_atol)
            group_dirs_raw = parser.get("input", "merged_group_dirs", fallback="")
            if group_dirs_raw.strip():
                self.merged_group_dirs = [
                    self.base_dir / line.strip()
                    for line in group_dirs_raw.splitlines()
                    if line.strip()
                ]

        if parser.has_section("output"):
            images_dir = parser.get("output", "images_path", fallback=str(self.images_path))
            self.images_path = self.base_dir / images_dir
            self.plot_start = parser.getint("output", "plot_start", fallback=self.plot_start)
            self.save_dpi = parser.getint("output", "save_dpi", fallback=self.save_dpi)

        self.apply_globals()
        return self

    def apply_globals(self) -> None:
        global dm, p_atm, num_exp, dt, N_time, MERGED_GROUP_DIRS, IMAGES_PATH, TARGET_X, X_INDEX_ATOL, PLOT_START, SAVE_DPI
        dm = self.dm
        p_atm = self.p_atm
        num_exp = self.num_exp
        dt = self.dt
        N_time = self.n_time
        MERGED_GROUP_DIRS = list(self.merged_group_dirs)
        IMAGES_PATH = self.images_path
        TARGET_X = self.target_x
        X_INDEX_ATOL = self.x_index_atol
        PLOT_START = self.plot_start
        SAVE_DPI = self.save_dpi

    def update_global_n_time(self, value: int | None) -> None:
        self.n_time = value
        global N_time
        N_time = value
