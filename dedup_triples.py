input_file = "Datasettriplets_VN07.tsv"
output_file = "Datasettriplets_VN07_dedup.tsv"

seen = set()
with open(input_file, "r", encoding="utf-8") as fin, open(output_file, "w", encoding="utf-8") as fout:
    for line in fin:
        parts = line.strip().split('\t')
        if len(parts) != 3:
            continue  # Bỏ qua dòng không đủ 3 phần
        triple = tuple(parts)
        if triple not in seen:
            fout.write(line)
            seen.add(triple)

print(f"Đã loại bỏ bộ ba trùng lặp và dòng không đủ 3 phần. File kết quả: {output_file}") 

# python dedup_triples.py