from __future__ import annotations

from pathlib import Path
import re
import pandas as pd
import numpy as np

# Допуск для группировки почти одинаковых X из разных файлов
X_ROUND_DIGITS = 12

BASE_DIR = Path(__file__).resolve().parent
SRC_ROOT = BASE_DIR / "linses" / "linses_src"
DST_ROOT = BASE_DIR / "linses" / "merged_parquet_full"

# Что и где искать
SERIES_MAP = {
    "base": ["NVIS", "DNVIS", "DVIS"],
    "0_helium_convex": ["MVIS", "DMVIS"],
    "1_helium_concave": ["MVIS", "DMVIS"],
    "2_freon_convex": ["MVIS", "DMVIS"],
    "3_freon_concave": ["MVIS", "DMVIS"],
}


def collect_series_files(folder: Path, prefix: str) -> list[Path]:
    """
    Собирает файлы вида PREFIX-0001 ... PREFIX-0900
    и сортирует их по номеру.
    """
    files = [
        p for p in folder.iterdir()
        if p.is_file() and p.name.startswith(prefix + "-")
    ]

    def extract_num(path: Path) -> int:
        m = re.search(r"-(\d+)$", path.name)
        return int(m.group(1)) if m else 10**9

    return sorted(files, key=extract_num)


def read_csv(csv_path: Path) -> pd.DataFrame:
    """
    Читает один исходный CSV-файл и нормализует имена колонок.
    """
    df = pd.read_csv(
        csv_path,
        header=0,
        skipinitialspace=True,
        dtype={
            "nodenumber": "int32",
            "x-coordinate": "float64",
            "y-coordinate": "float32",
            "pressure": "float32",
        },
        engine="c",
    )

    df.columns = [c.strip() for c in df.columns]
    required = {"nodenumber", "x-coordinate", "y-coordinate", "pressure"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"В файле {csv_path} нет колонок: {sorted(missing)}")

    return df[["nodenumber", "x-coordinate", "y-coordinate", "pressure"]].copy()


def normalize_x(series: pd.Series) -> pd.Series:
    """
    Округляет x так, чтобы одинаковые значения из разных файлов
    гарантированно попадали в одну группу.
    """
    return series.round(X_ROUND_DIGITS)


def format_x_for_index(x: float) -> str:
    return f"{x:.12e}"


def make_x_filename(x: float) -> str:
    x_str = format_x_for_index(x)
    safe = x_str.replace("+", "").replace("-", "m").replace(".", "_")
    return f"x_{safe}.parquet"


def convert_series(folder_name: str, prefix: str) -> None:
    src_dir = SRC_ROOT / folder_name
    dst_dir = DST_ROOT / folder_name / prefix
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = collect_series_files(src_dir, prefix)
    if not files:
        print(f"[SKIP] Нет файлов для {folder_name}/{prefix}")
        return

    parts: list[pd.DataFrame] = []

    for time_id, file_path in enumerate(files):
        df = read_csv(file_path)
        df["x-coordinate"] = normalize_x(df["x-coordinate"])
        df.insert(0, "time_id", time_id)
        parts.append(df)

        if (time_id + 1) % 100 == 0 or (time_id + 1) == len(files):
            print(f"{folder_name}/{prefix}: {time_id + 1}/{len(files)}")

    result = pd.concat(parts, ignore_index=True)

    index_rows: list[dict[str, str]] = []

    grouped = result.groupby("x-coordinate", sort=True, dropna=False)
    for x_value, group in grouped:
        filename = make_x_filename(float(x_value))
        out_path = dst_dir / filename

        out_df = group[["time_id", "nodenumber", "y-coordinate", "pressure"]].copy()
        out_df.to_parquet(out_path, index=False)

        index_rows.append({
            "x": format_x_for_index(float(x_value)),
            "filename": filename,
        })

    index_df = pd.DataFrame(index_rows)
    index_df.to_csv(dst_dir / "index.csv", index=False, encoding="utf-8", lineterminator="\n")

    print(f"[OK] {folder_name}/{prefix}")
    print(f"     x files: {len(index_rows)}")
    print(f"     index: {dst_dir / 'index.csv'}")



def main() -> None:
    for folder_name, prefixes in SERIES_MAP.items():
        for prefix in prefixes:
            convert_series(folder_name, prefix)


if __name__ == "__main__":
    main()
