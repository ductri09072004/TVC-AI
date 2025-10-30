#!/usr/bin/env python3
import argparse
import csv
import os
from typing import List


def read_csv(path: str, delimiter: str = ",", quotechar: str = '"') -> List[List[str]]:
    rows: List[List[str]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        for row in reader:
            if not row:
                continue
            rows.append(row)
    return rows


def write_csv(path: str, rows: List[List[str]], delimiter: str = ",", quotechar: str = '"') -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter, quotechar=quotechar)
        writer.writerows(rows)


def relabel(rows: List[List[str]], ok_label: str = "ok") -> List[List[str]]:
    if not rows:
        return rows
    header = rows[0]
    # Expect two columns: caption,label
    try:
        label_idx = header.index("label")
    except ValueError:
        # Fallback: last column as label
        label_idx = len(header) - 1

    out: List[List[str]] = [header]
    for row in rows[1:]:
        # Pad short rows
        if len(row) <= label_idx:
            row = row + [""] * (label_idx - len(row) + 1)
        label = (row[label_idx] or "").strip()
        new_label = "0" if label == ok_label else "1"
        new_row = list(row)
        new_row[label_idx] = new_label
        out.append(new_row)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Relabel CSV: ok->0, others->1")
    parser.add_argument("--input", type=str, default="data/dataset.csv", help="Input CSV path")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--output", type=str, help="Output CSV path (write relabeled file)")
    group.add_argument("--inplace", action="store_true", help="Overwrite input file with relabeled labels")
    parser.add_argument("--ok_label", type=str, default="ok", help="Label treated as 0 (default: ok)")
    parser.add_argument("--delimiter", type=str, default=",", help="CSV delimiter")
    parser.add_argument("--quotechar", type=str, default='"', help='CSV quote character')
    args = parser.parse_args()

    rows = read_csv(args.input, delimiter=args.delimiter, quotechar=args.quotechar)
    if not rows:
        print("Input is empty.")
        return

    out_rows = relabel(rows, ok_label=args.ok_label)

    target = args.input if args.inplace else (args.output or "data/dataset_binary.csv")
    write_csv(target, out_rows, delimiter=args.delimiter, quotechar=args.quotechar)
    print(f"Relabeled file written to: {target}")


if __name__ == "__main__":
    main()


