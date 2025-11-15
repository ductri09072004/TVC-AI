# Các Thay Đổi Để Giảm Overfitting

## ✅ Đã Sửa

### 1. **Tăng Dropout** (0.1 → 0.3)
- Thêm `--dropout` argument (default: 0.3)
- Áp dụng cho: `hidden_dropout_prob`, `attention_probs_dropout_prob`, `classifier_dropout`
- **Tác dụng:** Model không học thuộc lòng, phải học quy tắc thật

### 2. **Tăng Weight Decay** (0.01 → 0.1)
- Thêm `--weight_decay` argument (default: 0.1)
- Tăng regularization → Model không overfit
- **Tác dụng:** Giảm overfitting, model học quy tắc tổng quát hơn

### 3. **Gradient Clipping**
- Thêm `--max_grad_norm` argument (default: 1.0)
- Clip gradient để training ổn định
- **Tác dụng:** Tránh gradient explosion, training ổn định hơn

### 4. **Tăng Label Smoothing** (0.1 → 0.2)
- Default label smoothing tăng từ 0.1 → 0.2
- **Tác dụng:** Model không quá tự tin → Học quy tắc thật thay vì học thuộc

### 5. **Early Stopping Dựa Trên Dev Loss**
- Thay vì dùng train loss, giờ dùng **dev loss** để early stopping
- **Tác dụng:** Phát hiện overfitting sớm hơn (train loss giảm nhưng dev loss tăng)

### 6. **Tăng Patience** (3 → 5)
- Tăng patience từ 3 → 5 epochs
- **Tác dụng:** Model có thời gian học kỹ hơn trước khi dừng

### 7. **Giảm Min Delta** (0.001 → 0.0001)
- Giảm min_delta để nhạy hơn với cải thiện nhỏ
- **Tác dụng:** Phát hiện cải thiện tốt hơn

---

## 📊 So Sánh Trước/Sau

| Tham số | Trước | Sau | Lý do |
|---------|-------|-----|-------|
| Dropout | 0.1 (default) | 0.3 | Giảm overfitting |
| Weight Decay | 0.01 | 0.1 | Tăng regularization |
| Label Smoothing | 0.1 | 0.2 | Model không quá tự tin |
| Patience | 3 | 5 | Học kỹ hơn |
| Early Stopping | Train loss | **Dev loss** | Phát hiện overfitting tốt hơn |
| Gradient Clipping | Không có | 1.0 | Training ổn định |

---

## 🚀 Cách Sử Dụng

### Training với các tham số mới (khuyến nghị):

```bash
python run_bert_triple_classifier_phobert.py \
  --data_dir ./dataset \
  --bert_model vinai/phobert-base \
  --task_name kg \
  --output_dir output_moderation \
  --do_train \
  --do_eval \
  --tune_threshold \
  --loss_type label_smoothing \
  --label_smoothing 0.2 \
  --dropout 0.3 \
  --weight_decay 0.1 \
  --max_grad_norm 1.0 \
  --patience 5 \
  --learning_rate 2e-5 \
  --num_train_epochs 10 \
  --train_batch_size 32 \
  --eval_batch_size 8
```

### Tùy chỉnh thêm (nếu vẫn overfit):

```bash
# Tăng dropout hơn nữa
--dropout 0.4

# Tăng weight decay
--weight_decay 0.15

# Tăng label smoothing
--label_smoothing 0.3

# Giảm learning rate
--learning_rate 1e-5
```

---

## 🎯 Kỳ Vọng

Sau khi áp dụng các thay đổi này:

1. **Eval accuracy sẽ giảm** (từ 99% → khoảng 85-95%)
   - ✅ Đây là **TỐT** - Model không học thuộc nữa
   - Model học quy tắc thật thay vì học thuộc pattern

2. **Gap giữa train và dev loss nhỏ hơn**
   - Train loss và dev loss gần nhau → Không overfit

3. **Model generalize tốt hơn**
   - Dự đoán đúng trên dữ liệu thực tế
   - Không còn false positive nhiều như trước

---

## ⚠️ Lưu Ý

- **Eval accuracy giảm là BÌNH THƯỜNG** - Đây là dấu hiệu model không học thuộc
- Quan trọng là **dev loss** và **test accuracy** trên dữ liệu thực tế
- Nếu vẫn overfit, tăng thêm dropout/weight_decay
- Nếu underfit (accuracy quá thấp), giảm dropout/weight_decay

---

## 📝 Checklist

- [x] Thêm dropout config
- [x] Tăng weight decay
- [x] Thêm gradient clipping
- [x] Tăng label smoothing
- [x] Early stopping dựa trên dev loss
- [x] Tăng patience
- [ ] Test training với config mới
- [ ] So sánh kết quả trước/sau

