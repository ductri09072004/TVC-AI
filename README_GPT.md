# GPT-based Triple Classification

Script này sử dụng OpenAI GPT API để thực hiện triple classification thay vì fine-tuning BERT. Nó đọc cùng định dạng dữ liệu như phiên bản BERT và tạo ra kết quả đánh giá tương tự.

## Cài đặt

1. Cài đặt các thư viện cần thiết:
```bash
pip install -r requirements_gpt.txt
```

2. Lấy OpenAI API key từ [OpenAI Platform](https://platform.openai.com/api-keys)

## Sử dụng

### Cú pháp cơ bản:
```bash
python run_gpt_triple_classifier.py \
    --data_dir ./data/WN11-10 \
    --output_dir ./output_gpt_WN11_10 \
    --api_key YOUR_OPENAI_API_KEY \
    --do_eval \
    --do_predict
```

### Các tham số chính:

**Bắt buộc:**
- `--data_dir`: Đường dẫn đến thư mục chứa dữ liệu (train.tsv, dev.tsv, test.tsv)
- `--output_dir`: Thư mục lưu kết quả
- `--api_key`: OpenAI API key

**Tùy chọn:**
- `--model`: GPT model sử dụng (mặc định: "gpt-3.5-turbo")
- `--do_train`: Chạy đánh giá trên tập train
- `--do_eval`: Chạy đánh giá trên tập dev
- `--do_predict`: Chạy đánh giá trên tập test
- `--max_train_examples`: Số lượng tối đa examples train (để test nhanh)
- `--max_eval_examples`: Số lượng tối đa examples eval (để test nhanh)
- `--batch_size`: Kích thước batch cho API calls (mặc định: 10)
- `--seed`: Random seed (mặc định: 42)

### Ví dụ sử dụng:

1. **Chạy đánh giá trên tập dev:**
```bash
python run_gpt_triple_classifier.py \
    --data_dir ./data/WN11-10 \
    --output_dir ./output_gpt_WN11_10 \
    --api_key sk-your-api-key-here \
    --do_eval \
    --max_eval_examples 100
```

2. **Chạy đánh giá trên tập test:**
```bash
python run_gpt_triple_classifier.py \
    --data_dir ./data/WN11-10 \
    --output_dir ./output_gpt_WN11_10 \
    --api_key sk-your-api-key-here \
    --do_predict \
    --max_eval_examples 100
```

3. **Chạy đầy đủ (train, eval, predict):**
```bash
python run_gpt_triple_classifier.py \
    --data_dir ./data/WN11-10 \
    --output_dir ./output_gpt_WN11_10 \
    --api_key sk-your-api-key-here \
    --do_train \
    --do_eval \
    --do_predict \
    --max_train_examples 1000 \
    --max_eval_examples 100
```

## Kết quả

Script sẽ tạo ra các file sau trong thư mục output:

- `eval_results.txt`: Kết quả đánh giá trên tập dev (tương tự như BERT version)
- `test_results.txt`: Kết quả đánh giá trên tập test
- `eval_predictions.tsv`: Dự đoán chi tiết cho tập dev
- `test_predictions.tsv`: Dự đoán chi tiết cho tập test

### Format của eval_results.txt:
```
eval_accuracy = 0.8500
precision = 0.8600
recall = 0.8500
f1_score = 0.8550
```

## Lưu ý quan trọng

1. **Chi phí API**: Mỗi lần gọi API sẽ tốn phí. Hãy sử dụng `--max_eval_examples` để test với số lượng nhỏ trước.

2. **Rate limiting**: Script có built-in delay để tránh rate limiting. Có thể điều chỉnh `batch_size` nếu cần.

3. **Độ tin cậy**: GPT API có thể không ổn định như BERT fine-tuning. Script có retry mechanism để xử lý lỗi.

4. **Thời gian**: Chạy GPT API sẽ chậm hơn BERT local vì phải gọi API qua mạng.

## So sánh với BERT version

| Tính năng | BERT Version | GPT Version |
|-----------|--------------|-------------|
| Tốc độ | Nhanh (local) | Chậm (API calls) |
| Chi phí | Miễn phí | Tốn phí API |
| Độ chính xác | Phụ thuộc vào fine-tuning | Phụ thuộc vào GPT model |
| Khả năng mở rộng | Cần GPU | Không cần GPU |
| Dễ sử dụng | Phức tạp hơn | Đơn giản hơn |

## Troubleshooting

1. **API key không hợp lệ**: Kiểm tra lại API key
2. **Rate limiting**: Giảm `batch_size` hoặc tăng delay
3. **Out of memory**: Không áp dụng cho GPT version (chạy trên cloud)
4. **Kết quả không ổn định**: Thử với `temperature=0.0` (đã set mặc định) 