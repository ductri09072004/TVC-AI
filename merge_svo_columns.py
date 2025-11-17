"""
Script to merge the first N columns of a CSV (e.g., subject/verb/object) into a
single text column, separated by spaces. Useful for converting datasets like
`s,v,o,label` into `text,label`.
"""

import argparse
import csv
from pathlib import Path
from typing import Iterable, List


def merge_columns(
    input_path: Path,
    output_path: Path,
    columns_to_merge: int = 3,
    delimiter: str = ",",
    encoding: str = "utf-8",
) -> None:
    """
    Read a CSV file, merge the first `columns_to_merge` columns into one text field,
    and write the result to a new file.
    """
    if columns_to_merge < 1:
        raise ValueError("columns_to_merge phải >= 1")

    with input_path.open("r", encoding=encoding, newline="") as infile, output_path.open(
        "w", encoding=encoding, newline=""
    ) as outfile:
        reader = csv.reader(infile, delimiter=delimiter)
        writer = csv.writer(outfile, delimiter=delimiter)

        for row_index, row in enumerate(reader):
            if not row:
                writer.writerow(row)
                continue

            if row_index == 0 and any(cell.lower() in {"s", "subject"} for cell in row[:columns_to_merge]):
                # treat header row specially
                merged_header = "text"
                new_row = [merged_header] + row[columns_to_merge:]
                writer.writerow(new_row)
                continue

            if len(row) < columns_to_merge:
                merged_text = " ".join(cell.strip() for cell in row)
                writer.writerow([merged_text])
                continue

            merged_text = " ".join(cell.strip() for cell in row[:columns_to_merge] if cell.strip())
            remaining = row[columns_to_merge:]
            writer.writerow([merged_text, *remaining])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gộp N cột đầu tiên của file CSV thành một cột văn bản.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("input", help="Đường dẫn file CSV gốc (ví dụ: data.csv)")
    parser.add_argument(
        "-o",
        "--output",
        help="Đường dẫn file CSV đầu ra (mặc định: ghi đè lên file cũ)",
    )
    parser.add_argument(
        "-n",
        "--num-cols",
        type=int,
        default=3,
        help="Số cột đầu tiên cần gộp (ví dụ S,V,O => 3)",
    )
    parser.add_argument(
        "--delimiter",
        default=",",
        help="Ký tự phân tách cột trong CSV",
    )
    parser.add_argument(
        "--encoding",
        default="utf-8",
        help="Bảng mã file CSV",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = Path(args.input)
    if not input_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    output_path = Path(args.output) if args.output else input_path

    merge_columns(
        input_path=input_path,
        output_path=output_path,
        columns_to_merge=args.num_cols,
        delimiter=args.delimiter,
        encoding=args.encoding,
    )

    print(f"Đã gộp {args.num_cols} cột đầu tiên và lưu vào: {output_path}")


if __name__ == "__main__":
    main()

