#!/usr/bin/env python3
"""
Convert dataset.csv from 'caption,label' format to 's,v,o,label' format.
Uses heuristic patterns to extract Subject-Verb-Object from Vietnamese captions.
"""

import csv
import re
import sys
from typing import Tuple, Optional

# Common Vietnamese verbs (infinitive and conjugated forms)
COMMON_VERBS = [
    # Multi-word verbs first (longer first) - these have priority
    'xuất hiện', 'vật lộn', 'rượt đuổi', 'tạo dáng', 'tụ tập', 'giải tỏa',
    'xông vào', 'thì thầm', 'dựa vào', 'cho thấy', 'hiển thị', 'vuốt ve',
    'nằm gục',
    # Single-word verbs
    'nâng', 'uống', 'nói', 'quay', 'mô tả', 'cụng', 'cười', 'rót', 'khoác',
    'chụp', 'nằm', 'cầm', 'nhảy', 'khuyên', 'xô', 'đánh', 'tát', 'vật',
    'đấm', 'đe dọa', 'ném', 'mắng', 'đạp', 'đập', 'chặt', 'gào', 'phá',
    'lao', 'chĩa', 'liếm', 'uốn', 'cởi', 'vuốt', 'ngả', 'mặc', 'thử',
    'mua', 'bán', 'sử dụng', 'xem', 'nghe', 'ăn', 'chạy', 'đi', 'đứng',
    'ngồi', 'làm', 'là', 'có', 'được', 'nhấn', 'giới thiệu', 'quăng',
    'hét', 'đặt', 'khoe', 'vá', 'tràn', 'cổ vũ', 'kéo', 'ôm', 'hôn',
    'gõ', 'gọi', 'bảo', 'nhấp', 'liếc',
    # Added for cleaning malformed SVO lines
    'lia', 'giơ', 'hứa', 'tuyên bố', 'gội', 'gội đầu', 'nhận', 'hoàn',
    'đeo', 'trao', 'cảm ơn', 'khẳng định', 'sửa', 'giao', 'cúi',
    'nắm', 'gào thét', 'uốn'
]


def find_verb_in_text(text: str) -> Tuple[Optional[str], Optional[int], Optional[int]]:
    """Find the first main verb in text and return (verb, start_pos, end_pos).
    Prioritizes longer verbs (multi-word) and verbs before commas.
    """
    text_lower = text.lower()
    
    # Sort verbs by length (longer first) to match multi-word verbs first
    sorted_verbs = sorted(COMMON_VERBS, key=len, reverse=True)
    
    # Find comma and conjunction positions (for complex sentences)
    comma_pos = text.find(',')
    if comma_pos == -1:
        comma_pos = len(text)
    
    # Find "và" (and) position - prefer verbs before conjunctions
    conj_pos = text_lower.find(' và ')
    if conj_pos == -1:
        conj_pos = len(text)
    else:
        conj_pos += 1  # Position after "và"
    
    # Use the earlier of comma or conjunction
    clause_end = min(comma_pos, conj_pos)
    
    # Collect all matches first, then choose best
    # Use dict to track best match at each position (prefer longer verbs at same position)
    matches_by_pos = {}  # start_pos -> (len, start_pos, verb_original, start_pos, end_pos)
    
    for verb in sorted_verbs:
        verb_lower = verb.lower()
        # Match verb at word boundary
        pattern = r'\b' + re.escape(verb_lower) + r'\b'
        match = re.search(pattern, text_lower)
        if match:
            start_pos = match.start()
            # Prefer verbs before comma/conjunction (in main clause)
            if start_pos < clause_end or clause_end == len(text):
                end_pos = match.end()
                verb_original = text[start_pos:end_pos]
                # If we already have a match at this position, keep the longer one
                if start_pos not in matches_by_pos or len(verb) > matches_by_pos[start_pos][0]:
                    matches_by_pos[start_pos] = (len(verb), start_pos, verb_original, start_pos, end_pos)
    
    all_matches = list(matches_by_pos.values())
    
    if all_matches:
        # First, sort by position to ensure we process in order
        all_matches.sort(key=lambda x: x[1])  # Sort by position first
        
        # Check for compound verbs (verbs close together like "ôm hôn")
        # Mark later verbs in compounds as lower priority
        for i in range(len(all_matches)):
            len1, pos1, verb1, _, _ = all_matches[i]
            for j in range(i+1, len(all_matches)):
                len2, pos2, verb2, _, _ = all_matches[j]
                # If two verbs are very close (within 10 chars), prefer the first one
                if 0 < (pos2 - pos1) < 10:
                    # Mark second verb as compound by making its length negative
                    all_matches[j] = (-abs(len2), pos2, verb2, all_matches[j][3], all_matches[j][4])
        
        # Now sort by: 1) length (longer first, but negative = compound = much lower), 2) position (earlier first)
        # Format: (length, start_pos, verb_original, start_pos, end_pos)
        all_matches.sort(key=lambda x: (
            -abs(x[0]) if x[0] > 0 else abs(x[0]) - 10000,  # Length: longer first, negative = much lower
            x[1]  # Position: earlier first (when same length)
        ))
        _, _, verb_original, start_pos, end_pos = all_matches[0]
        return (verb_original, start_pos, end_pos)
    
    return (None, None, None)


def extract_svo(text: str) -> Tuple[str, str, str]:
    """Extract Subject, Verb, Object from Vietnamese sentence using simple heuristics."""
    text = text.strip()
    if not text:
        return ("", "", "")
    
    # Remove quotes if present
    text = text.strip('"\'')
    
    # Find verb position
    verb, verb_start, verb_end = find_verb_in_text(text)
    
    if verb is None or verb_start is None:
        # No verb found, try simple patterns
        # Pattern: "X là Y" or "X làm Y"
        simple_patterns = [
            r'^(.+?)\s+(là|với|của)\s+(.+)$',
        ]
        
        for pattern in simple_patterns:
            match = re.match(pattern, text, re.IGNORECASE)
            if match:
                s = match.group(1).strip()
                v = match.group(2).strip()
                o = match.group(3).strip()
                return (s, v, o)
        
        # Fallback: try to split by common patterns
        # Pattern: "X Y Z" where Y might be a verb we missed
        # Try to find first meaningful word as potential subject
        words = text.split()
        if len(words) >= 3:
            # Take first 2-3 words as subject, rest as object
            # This handles cases like "Người phụ nữ kéo dây..."
            potential_subject = ' '.join(words[:min(3, len(words))])
            potential_object = ' '.join(words[min(3, len(words)):])
            return (potential_subject, "", potential_object)
        
        # Last resort: treat whole as object
        return ("", "", text)
    
    # Split text around verb
    subject = text[:verb_start].strip()
    verb_word = verb
    object_text = text[verb_end:].strip()
    
    # Clean up spaces
    subject = ' '.join(subject.split())
    verb_word = verb_word.strip()
    object_text = ' '.join(object_text.split())
    
    # If subject is empty, try to get from first few words
    if not subject:
        words = text[:verb_start].split()
        if words:
            subject = ' '.join(words)
    
    return (subject, verb_word, object_text)


def convert_dataset(input_file: str, output_file: str):
    """Convert dataset from caption,label to s,v,o,label format."""
    converted = 0
    errors = 0
    
    with open(input_file, 'r', encoding='utf-8') as f_in, \
         open(output_file, 'w', encoding='utf-8', newline='') as f_out:
        
        reader = csv.DictReader(f_in)
        writer = csv.writer(f_out, quoting=csv.QUOTE_ALL)
        
        # Write header
        writer.writerow(['s', 'v', 'o', 'label'])
        
        for row in reader:
            caption = row.get('caption', '').strip()
            label = row.get('label', '').strip()
            
            if not caption:
                continue
            
            try:
                s, v, o = extract_svo(caption)
                
                # Write to output
                writer.writerow([s, v, o, label])
                converted += 1
                
                if converted % 1000 == 0:
                    print(f"Đã xử lý {converted} dòng...", file=sys.stderr)
                    
            except Exception as e:
                errors += 1
                print(f"Lỗi ở dòng {converted + errors}: {e}", file=sys.stderr)
                # Write original caption as object, empty subject/verb
                writer.writerow(["", "", caption, label])
    
    print(f"\nHoàn thành!")
    print(f"- Đã chuyển đổi: {converted} dòng")
    print(f"- Lỗi: {errors} dòng")
    print(f"- File output: {output_file}")
    print(f"\n⚠️  Lưu ý: Kết quả tách SVO tự động có thể không chính xác 100%.")
    print(f"   Vui lòng review và chỉnh sửa thủ công nếu cần.")


if __name__ == "__main__":
    input_file = "data/dataset.csv"
    output_file = "data/dataset_svo.csv"
    
    if len(sys.argv) > 1:
        input_file = sys.argv[1]
    if len(sys.argv) > 2:
        output_file = sys.argv[2]
    
    print(f"Chuyển đổi: {input_file} -> {output_file}")
    convert_dataset(input_file, output_file)

