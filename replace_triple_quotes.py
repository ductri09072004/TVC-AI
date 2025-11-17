'''
Utility script to replace triple double-quotes (""") with a single double-quote (")
in CSV or text files. Use this when a dataset contains fields wrapped with triple
quotes and you need to normalize them to standard CSV quoting.
'''

import argparse
from pathlib import Path


def replace_triple_quotes(file_path: Path, dry_run: bool = False) -> int:
    """
    Replace occurrences of triple double-quotes with a single double-quote.

    Args:
        file_path: Path to the file to modify.
        dry_run: If True, only report the number of replacements without writing.

    Returns:
        int: Number of replacements performed (or would be performed in dry-run).
    """
    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy file: {file_path}")

    original_text = file_path.read_text(encoding="utf-8")
    new_text = original_text.replace('"""', '"')

    replacements = (len(original_text) - len(new_text)) // (len('"""') - len('"'))

    if replacements > 0 and not dry_run:
        file_path.write_text(new_text, encoding="utf-8")

    return replacements


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Thay thế triple quotes (\"\"\") bằng double quotes (\") trong file."
    )
    parser.add_argument("file", help="Đường dẫn tới file cần xử lý (ví dụ: data.csv)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Chỉ thống kê số lượng thay thế, không ghi file.",
    )
    args = parser.parse_args()

    file_path = Path(args.file)

    try:
        replacements = replace_triple_quotes(file_path, dry_run=args.dry_run)
    except FileNotFoundError as exc:
        print(exc)
        return

    if replacements == 0:
        print(f"Không tìm thấy triple quotes trong file: {file_path}")
    else:
        if args.dry_run:
            print(f"(Dry-run) Sẽ thay thế {replacements} triple quotes trong {file_path}")
        else:
            print(f"Đã thay thế {replacements} triple quotes trong {file_path}")


if __name__ == "__main__":
    main()

