#!/usr/bin/env python3
import argparse
import csv
from typing import List, Tuple, Set


def read_csv_rows(path: str, delimiter: str, quotechar: str) -> List[Tuple[int, List[str]]]:
    rows: List[Tuple[int, List[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        for idx, row in enumerate(reader, start=1):
            rows.append((idx, row))
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Check CSV for missing/invalid labels.")
    parser.add_argument("path", type=str, help="Path to CSV file (e.g., data/dataset.csv)")
    parser.add_argument("--delimiter", type=str, default=",", help="CSV delimiter (default: ,)")
    parser.add_argument("--quotechar", type=str, default='"', help='CSV quote character (default: ")')
    parser.add_argument("--has_header", action="store_true", help="Indicate the first row is header")
    parser.add_argument("--text_col", type=int, default=0, help="Index of caption/text column (default: 0)")
    parser.add_argument("--label_col", type=int, default=1, help="Index of label column (default: 1)")
    parser.add_argument(
        "--valid_labels",
        type=str,
        default="0,1",
        help="Comma-separated list of valid labels (default: 0,1)",
    )
    parser.add_argument(
        "--report",
        type=str,
        default="",
        help="Optional path to write a CSV report of problematic rows",
    )

    args = parser.parse_args()

    rows = read_csv_rows(args.path, args.delimiter, args.quotechar)
    if args.has_header and rows:
        header = rows.pop(0)[1]
    else:
        header = []

    valid: Set[str] = {v.strip() for v in args.valid_labels.split(",") if v.strip() != ""}

    missing: List[Tuple[int, List[str], str]] = []  # (line_no, row, reason)
    invalid: List[Tuple[int, List[str], str]] = []

    for line_no, row in rows:
        # Defensive: pad row length
        label = ""
        if args.label_col < len(row):
            label = (row[args.label_col] or "").strip()

        if label == "":
            missing.append((line_no, row, "empty label"))
            continue

        # Normalize numeric labels like "0.0" -> "0"
        norm = label
        try:
            norm = str(int(float(label)))
        except Exception:
            pass

        if norm not in valid:
            invalid.append((line_no, row, f"invalid label '{label}' (normalized: '{norm}')"))

    total = len(rows)
    print(f"Checked {total} data rows (excluding header={args.has_header}).")
    print(f"Missing labels: {len(missing)} | Invalid labels: {len(invalid)} | Valid labels: {total - len(missing) - len(invalid)}")

    if missing or invalid:
        print("\nSamples:")
        for grp_name, grp in [("Missing", missing[:10]), ("Invalid", invalid[:10])]:
            if not grp:
                continue
            print(f"- {grp_name} (showing up to 10):")
            for line_no, row, reason in grp:
                preview = row[args.text_col] if args.text_col < len(row) else ""
                if len(preview) > 120:
                    preview = preview[:117] + "..."
                print(f"  line {line_no}: {reason} | text='{preview}'")

    if args.report:
        with open(args.report, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=args.delimiter, quotechar=args.quotechar)
            if header:
                writer.writerow(header + ["__issue__"])
            for line_no, row, reason in missing + invalid:
                writer.writerow(row + [reason])
        print(f"Report written to: {args.report}")


if __name__ == "__main__":
    main()


