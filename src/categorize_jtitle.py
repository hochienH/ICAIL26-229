import os
import csv
import json
import random
from pathlib import Path

# Dynamic Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'first_instance_case')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'categorized_case')

# 定義分類規則
# 結構: Category -> { Target Group Name: [Keywords list] }
RULES = {
    '民事': {
        '清償借款_清償債務': ['清償借款', '清償債務'],
        '損害賠償': ['損害賠償'],
        '離婚': ['離婚'],
        '分割共有物': ['分割共有物']
    },
    '刑事': {
        '毒品危害防制條例': ['毒品危害防制條例'],
        '竊盜_詐欺': ['竊盜', '詐欺'],
        '公共危險': ['公共危險'],
        '傷害': ['傷害']
    },
    '行政': {
        '交通裁決': ['交通裁決'],
        '稅': ['稅']
    }
}

def match_group(category, jtitle):
    """
    根據 category 和 jtitle 判斷所屬的分類群組
    """
    if category not in RULES:
        return None
        
    category_rules = RULES[category]
    jtitle = jtitle.strip()
    
    for group_name, keywords in category_rules.items():
        for keyword in keywords:
            if keyword in jtitle:
                return group_name
                
    return None

def categorize_cases():
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory not found: {INPUT_DIR}")
        return

    # 確保輸出目錄存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    # 用來儲存分類後的結果
    # 結構: filename -> list of rows
    categorized_data = {}
    
    # 預先初始化所有可能的輸出檔案列表
    for cat, groups in RULES.items():
        for group in groups:
            filename = f"{cat}_{group}.csv"
            categorized_data[filename] = []

    # 讀取輸入目錄下的所有 CSV
    all_files = [f for f in os.listdir(INPUT_DIR) if f.endswith('.csv')]
    
    # 過濾 2016 年以前的檔案 (含 2016)
    input_files = []
    for filename in all_files:
        # 嘗試從檔名解析年份 (YYYYMM.csv)
        try:
            # 取檔名前 4 碼作為年份
            if filename[:4].isdigit():
                year = int(filename[:4])
                if year <= 2020:
                    input_files.append(filename)
        except:
            continue

    print(f"Found {len(all_files)} files, processing {len(input_files)} files (Year <= 2020)...")
    
    total_processed = 0
    total_matched = 0
    
    for filename in input_files:
        filepath = os.path.join(INPUT_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                
                if 'category' not in reader.fieldnames or 'JTITLE' not in reader.fieldnames:
                    print(f"Warning: File {filename} format incorrect, skipping")
                    continue
                    
                for row in reader:
                    total_processed += 1
                    category = row['category']
                    jtitle = row.get('JTITLE', '')
                    
                    # 判斷是否符合任何規則
                    group_name = match_group(category, jtitle)
                    
                    if group_name:
                        # 構建輸出檔名
                        output_filename = f"{category}_{group_name}.csv"
                        row['source_file'] = filename
                        categorized_data[output_filename].append(row)
                        total_matched += 1
                        
        except Exception as e:
            print(f"Error processing {filename}: {e}")

    print(f"Classification complete. Processed {total_processed}, Matched {total_matched}.")
    print(f"Writing results to {OUTPUT_DIR} ...")
    
    # 寫入分類後的檔案
    for filename, rows in categorized_data.items():
        if not rows:
            continue
            
        # 進行抽樣：如果數量大於 30000，隨機抽取 30000 筆
        if len(rows) > 30000:
            rows = random.sample(rows, 30000)
            
        # 修改輸出檔名，加上 _30000
        name_stem = os.path.splitext(filename)[0]
        output_filename = f"{name_stem}_30000.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        fieldnames = ['檔名', 'JTITLE', 'category', 'source_file']
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows:
                    writer.writerow({k: row.get(k, '') for k in fieldnames})
            print(f"  Created: {filename} ({len(rows)} rows)")
        except Exception as e:
            print(f"Error writing {filename}: {e}")

if __name__ == "__main__":
    categorize_cases()
