#!/usr/bin/env python3
import argparse
import csv
import os
import random
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
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter=delimiter, quotechar=quotechar)
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Split dataset.csv into train/dev/test 80/10/10")
    parser.add_argument("--input", type=str, default="data/dataset.csv", help="Input CSV path")
    parser.add_argument("--out_dir", type=str, default="dataset", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for shuffling")
    parser.add_argument("--delimiter", type=str, default=",", help="CSV delimiter")
    parser.add_argument("--quotechar", type=str, default='"', help='CSV quote character')
    args = parser.parse_args()

    rows = read_csv(args.input, delimiter=args.delimiter, quotechar=args.quotechar)
    if not rows:
        print("Input is empty.")
        return

    header, data = rows[0], rows[1:]

    random.seed(args.seed)
    random.shuffle(data)

    n = len(data)
    n_train = int(0.8 * n)
    n_dev = int(0.1 * n)
    n_test = n - n_train - n_dev

    train_rows = [header] + data[:n_train]
    dev_rows = [header] + data[n_train:n_train + n_dev]
    test_rows = [header] + data[n_train + n_dev:]

    write_csv(os.path.join(args.out_dir, "train.csv"), train_rows, delimiter=args.delimiter, quotechar=args.quotechar)
    write_csv(os.path.join(args.out_dir, "dev.csv"), dev_rows, delimiter=args.delimiter, quotechar=args.quotechar)
    write_csv(os.path.join(args.out_dir, "test.csv"), test_rows, delimiter=args.delimiter, quotechar=args.quotechar)

    print(f"Wrote {len(train_rows)-1} train, {len(dev_rows)-1} dev, {len(test_rows)-1} test rows to '{args.out_dir}'")


if __name__ == "__main__":
    main()


