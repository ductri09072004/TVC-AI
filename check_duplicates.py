#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple


def read_csv_rows(path: str, delimiter: str = ",", quotechar: str = '"') -> List[Tuple[int, List[str]]]:
    rows: List[Tuple[int, List[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        for idx, row in enumerate(reader, start=1):
            # Skip empty lines
            if not row or all(cell.strip() == "" for cell in row):
                continue
            rows.append((idx, row))
    return rows


def key_from_row(row: List[str], mode: str) -> str:
    if mode == "text":
        # Use first column as text; robust to leading/trailing spaces
        text = (row[0] if len(row) > 0 else "").strip()
        return text
    elif mode == "row":
        # Full-row key (exact duplicate across all columns)
        return "\u0001".join(cell.strip() for cell in row)
    else:
        raise ValueError("Unknown mode: %s" % mode)


def find_duplicates(rows: List[Tuple[int, List[str]]], mode: str) -> Dict[str, List[int]]:
    key_to_lines: Dict[str, List[int]] = defaultdict(list)
    for line_no, row in rows:
        k = key_from_row(row, mode)
        key_to_lines[k].append(line_no)
    return {k: v for k, v in key_to_lines.items() if len(v) > 1 and k != ""}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check duplicate entries in a CSV file.")
    parser.add_argument("path", type=str, help="Path to CSV file (e.g., data/dataset.csv)")
    parser.add_argument(
        "--mode",
        choices=["text", "row"],
        default="text",
        help="Duplicate check mode: 'text' (first column only) or 'row' (entire row match)",
    )
    parser.add_argument(
        "--delimiter",
        type=str,
        default=",",
        help="CSV delimiter (default: ,)",
    )
    parser.add_argument(
        "--quotechar",
        type=str,
        default='"',
        help='CSV quote character (default: ")',
    )
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument(
        "--output",
        type=str,
        help="Path to write a de-duplicated CSV (keeps first occurrence)",
    )
    out_group.add_argument(
        "--inplace",
        action="store_true",
        help="Overwrite the input file with a de-duplicated version (keeps first occurrence)",
    )

    args = parser.parse_args()

    rows = read_csv_rows(args.path, delimiter=args.delimiter, quotechar=args.quotechar)
    duplicates = find_duplicates(rows, mode=args.mode)

    total = sum(len(v) for v in duplicates.values())
    unique_keys = len(duplicates)

    if not duplicates:
        print("No duplicates found.")
        # Even if no duplicates, still support writing out a copy if requested
        if args.output or args.inplace:
            target_path = args.path if args.inplace else args.output
            with open(target_path, "w", encoding="utf-8", newline="") as f:
                writer = csv.writer(f, delimiter=args.delimiter, quotechar=args.quotechar)
                for _, row in rows:
                    writer.writerow(row)
            print(f"Written (unchanged) file to: {target_path}")
        return

    print(f"Found {unique_keys} duplicate keys affecting {total} rows (mode={args.mode}).")
    print("")
    for key, line_numbers in sorted(duplicates.items(), key=lambda x: (len(x[1]), x[0]), reverse=True):
        sample_preview = key.replace("\n", " ")
        if len(sample_preview) > 120:
            sample_preview = sample_preview[:117] + "..."
        print("- Key:", sample_preview)
        print("  Lines:", ", ".join(str(n) for n in line_numbers))
        print(f"  Count: {len(line_numbers)}")
        print("")

    # Optionally write de-duplicated file
    if args.output or args.inplace:
        seen = set()
        deduped: List[List[str]] = []
        kept = 0
        removed = 0
        for _, row in rows:
            k = key_from_row(row, args.mode)
            if k == "":
                # keep empty-key rows as-is
                deduped.append(row)
                kept += 1
                continue
            if k in seen:
                removed += 1
                continue
            seen.add(k)
            deduped.append(row)
            kept += 1

        target_path = args.path if args.inplace else args.output
        with open(target_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=args.delimiter, quotechar=args.quotechar)
            for row in deduped:
                writer.writerow(row)
        print(f"De-duplicated file written to: {target_path}")
        print(f"Kept: {kept} rows; Removed duplicates: {removed} rows (mode={args.mode}).")


if __name__ == "__main__":
    main()


