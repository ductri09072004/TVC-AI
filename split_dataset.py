"""
Split a CSV dataset into train and test sets (default 70/30).

By default, the script writes:
    dataset/<ten_file>/train.csv
    dataset/<ten_file>/test.csv

You can override the destinations with --train-out / --test-out if needed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Tuple

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split a CSV dataset into train/test subsets.")
    parser.add_argument("--input", "-i", required=True, help="Đường dẫn tới file CSV nguồn.")
    parser.add_argument(
        "--train-out",
        "-tr",
        help="Đường dẫn file train.csv. Nếu bỏ trống sẽ dùng dataset/<ten_file>/train.csv.",
    )
    parser.add_argument(
        "--test-out",
        "-te",
        help="Đường dẫn file test.csv. Nếu bỏ trống sẽ dùng dataset/<ten_file>/test.csv.",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        default="dataset",
        help="Thư mục gốc khi tự động tạo đường dẫn (mặc định: %(default)s).",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.3,
        help="Tỷ lệ dữ liệu dành cho test (0 < test-size < 1). Mặc định: %(default)s.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed khi shuffle.")
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Shuffle dữ liệu trước khi tách (khuyên dùng).",
    )
    return parser.parse_args()


def resolve_output_paths(
    input_path: Path, train_out: str | None, test_out: str | None, output_dir: str
) -> Tuple[Path, Path]:
    base_dir = Path(output_dir).expanduser().resolve()
    stem_dir = base_dir / input_path.stem

    train_path = Path(train_out).resolve() if train_out else (stem_dir / "train.csv")
    test_path = Path(test_out).resolve() if test_out else (stem_dir / "test.csv")

    for path in (train_path.parent, test_path.parent):
        path.mkdir(parents=True, exist_ok=True)

    return train_path, test_path


def split_dataset(df: pd.DataFrame, test_size: float, shuffle: bool, seed: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    if not 0.0 < test_size < 1.0:
        raise ValueError("test-size phải nằm trong khoảng (0, 1).")

    if shuffle:
        df = df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    test_count = int(len(df) * test_size)
    test_df = df.iloc[:test_count].reset_index(drop=True)
    train_df = df.iloc[test_count:].reset_index(drop=True)

    return train_df, test_df


def main() -> None:
    args = parse_args()
    try:
        input_path = Path(args.input).expanduser().resolve()
        if not input_path.exists():
            raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

        train_out_path, test_out_path = resolve_output_paths(input_path, args.train_out, args.test_out, args.output_dir)

        print(f"Đang tải dữ liệu từ {input_path} ...")
        df = pd.read_csv(input_path)
        print(f"Tổng số dòng: {len(df)}")

        train_df, test_df = split_dataset(df, args.test_size, args.shuffle, args.seed)
        print(f"Train: {len(train_df)} dòng | Test: {len(test_df)} dòng")

        train_df.to_csv(train_out_path, index=False)
        test_df.to_csv(test_out_path, index=False)
        print(f"Đã lưu train -> {train_out_path}")
        print(f"Đã lưu test  -> {test_out_path}")

    except Exception as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

