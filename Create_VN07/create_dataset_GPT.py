import openai
from tqdm import tqdm
import re
import os

# Lệnh chạy
# python ./Create_VN07/create_dataset_GPT.py
openai.api_key = "sk-proj-AuuTqAJjwLeYSOWJIINjT3BlbkFJJ11UiM3uH6VNmkKQm5GW"

def generate_triplet(prompt, model="gpt-4o"): #gpt-3.5-turbo
    client = openai.OpenAI(api_key=openai.api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "Bạn là một chuyên gia về du lịch, hãy tạo ra các bộ ba kiến thức về du lịch Việt Nam."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.7,
        max_tokens=80
    )
    return response.choices[0].message.content

def clean_triplet(line):
    # Loại bỏ số thứ tự, dấu gạch đầu dòng, dấu ngoặc, khoảng trắng thừa
    line = line.strip()
    line = re.sub(r"^[-•]\s*", "", line)  # Xóa gạch đầu dòng hoặc bullet
    line = re.sub(r"^\d+\.\s*", "", line)  # Xóa số thứ tự đầu dòng
    line = line.strip("()[]{} ")  # Xóa dấu ngoặc và khoảng trắng
    parts = [p.strip() for p in line.split(",")]
    if len(parts) == 3:
        return tuple(parts)
    # Nếu không đúng định dạng, thử tách bằng tab
    parts = [p.strip() for p in line.split("\t")]
    if len(parts) == 3:
        return tuple(parts)
    return None

def main(num_triplets=100, output_file="Datasettriplets_VN07.tsv"):
    batch_size = 30
    triplet_set = set()
    # Đọc các bộ ba cũ nếu file đã tồn tại
    if os.path.exists(output_file):
        with open(output_file, "r", encoding="utf-8") as f:
            for line in f:
                triplet = clean_triplet(line)
                if triplet:
                    triplet_set.add(triplet)
    pbar = tqdm(total=num_triplets, desc="Tạo bộ ba mới")
    added = 0
    while added < num_triplets:
        need = min(batch_size, num_triplets - added)
        prompt = (
            f"Hãy liệt kê {need} bộ ba (địa danh, quan hệ, đối tượng) về du lịch Việt Nam, mỗi bộ ba một dòng, không giải thích, không đánh số, không gạch đầu dòng, chỉ chọn các đặc trưng nổi bật, độc đáo."
        )
        triplets_text = generate_triplet(prompt)
        lines = [line for line in triplets_text.split('\n') if line.strip()]
        for line in lines:
            triplet = clean_triplet(line)
            if triplet and triplet not in triplet_set:
                triplet_set.add(triplet)
                added += 1
                pbar.update(1)
            if added >= num_triplets:
                break
    pbar.close()
    # Ghi nối tiếp các bộ ba mới vào file (không xóa bộ ba cũ)
    with open(output_file, "a", encoding="utf-8") as f:
        written = 0
        for triplet in list(triplet_set)[-num_triplets:]:
            f.write("\t".join(triplet) + "\n")
            written += 1
    print(f"Đã thêm {written} bộ ba mới vào {output_file}")

if __name__ == "__main__":
    main(num_triplets=1000, output_file="Datasettriplets_VN07.tsv")