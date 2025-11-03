#!/usr/bin/env python3
import argparse
import csv
from collections import defaultdict
from typing import Dict, List, Tuple, Set


def read_csv_rows(path: str, delimiter: str, quotechar: str) -> List[Tuple[int, List[str]]]:
    rows: List[Tuple[int, List[str]]] = []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar=quotechar)
        for idx, row in enumerate(reader, start=1):
            rows.append((idx, row))
    return rows


def key_from_row(row: List[str], mode: str, text_col: int = 0) -> str:
    if mode == "text":
        # Use text column as key; robust to leading/trailing spaces
        text = (row[text_col] if text_col < len(row) else "").strip()
        return text
    elif mode == "row":
        # Full-row key (exact duplicate across all columns)
        return "\u0001".join(cell.strip() for cell in row)
    else:
        raise ValueError("Unknown mode: %s" % mode)


def find_duplicates(rows: List[Tuple[int, List[str]]], mode: str, text_col: int = 0) -> Dict[str, List[int]]:
    key_to_lines: Dict[str, List[int]] = defaultdict(list)
    for line_no, row in rows:
        k = key_from_row(row, mode, text_col)
        key_to_lines[k].append(line_no)
    return {k: v for k, v in key_to_lines.items() if len(v) > 1 and k != ""}


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
    parser.add_argument(
        "--check_duplicates",
        action="store_true",
        help="Also check for duplicate entries",
    )
    parser.add_argument(
        "--dup_mode",
        choices=["text", "row"],
        default="text",
        help="Duplicate check mode: 'text' (text column only) or 'row' (entire row match) (default: text)",
    )
    out_group = parser.add_mutually_exclusive_group()
    out_group.add_argument(
        "--dedup_output",
        type=str,
        help="Path to write a de-duplicated CSV (keeps first occurrence, only used with --check_duplicates)",
    )
    out_group.add_argument(
        "--dedup_inplace",
        action="store_true",
        help="Overwrite the input file with a de-duplicated version (keeps first occurrence, only used with --check_duplicates)",
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

    # Check for duplicates if requested
    duplicates: Dict[str, List[int]] = {}
    if args.check_duplicates:
        duplicates = find_duplicates(rows, mode=args.dup_mode, text_col=args.text_col)

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
    
    if args.check_duplicates:
        dup_total = sum(len(v) for v in duplicates.values())
        dup_unique_keys = len(duplicates)
        if duplicates:
            print(f"Duplicates: {dup_unique_keys} duplicate keys affecting {dup_total} rows (mode={args.dup_mode})")
        else:
            print(f"Duplicates: None found (mode={args.dup_mode})")

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

    if args.check_duplicates and duplicates:
        print("\nDuplicate samples (showing up to 10):")
        for key, line_numbers in sorted(duplicates.items(), key=lambda x: (len(x[1]), x[0]), reverse=True)[:10]:
            sample_preview = key.replace("\n", " ")
            if len(sample_preview) > 120:
                sample_preview = sample_preview[:117] + "..."
            print(f"- Key: {sample_preview}")
            print(f"  Lines: {', '.join(str(n) for n in line_numbers)}")
            print(f"  Count: {len(line_numbers)}")

    if args.report:
        with open(args.report, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=args.delimiter, quotechar=args.quotechar)
            if header:
                writer.writerow(header + ["__issue__"])
            for line_no, row, reason in missing + invalid:
                writer.writerow(row + [reason])
        print(f"Report written to: {args.report}")

    # Optionally write de-duplicated file
    if args.check_duplicates and (args.dedup_output or args.dedup_inplace):
        seen = set()
        deduped: List[List[str]] = []
        kept = 0
        removed = 0
        for _, row in rows:
            k = key_from_row(row, args.dup_mode, args.text_col)
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

        target_path = args.path if args.dedup_inplace else args.dedup_output
        with open(target_path, "w", encoding="utf-8", newline="") as f:
            writer = csv.writer(f, delimiter=args.delimiter, quotechar=args.quotechar)
            if header:
                writer.writerow(header)
            for row in deduped:
                writer.writerow(row)
        print(f"\nDe-duplicated file written to: {target_path}")
        print(f"Kept: {kept} rows; Removed duplicates: {removed} rows (mode={args.dup_mode}).")


if __name__ == "__main__":
    main()


