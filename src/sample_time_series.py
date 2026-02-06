import os
import csv
import random
from pathlib import Path

# Dynamic Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_DIR = os.path.join(BASE_DIR, 'data', 'first_instance_case')
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'sampled_time_series')
SAMPLE_SIZE = 110000

def sample_continuous_cases():
    if not os.path.exists(INPUT_DIR):
        print(f"Input directory not found: {INPUT_DIR}")
        return

    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        
    print("Scanning and sorting files...")
    
    # 1. 獲取所有 CSV 檔案並按時間排序
    all_files = []
    for f in os.listdir(INPUT_DIR):
        if not f.endswith('.csv') or f in ['judgments_filtered.csv', 'civil_judgments_filtered.csv']:
            continue
        try:
            if f[:4].isdigit():
                file_date = int(f[:6]) # YYYYMM
                all_files.append((file_date, f))
        except:
            continue
            
    # 按年月排序 (從小到大)
    all_files.sort(key=lambda x: x[0])
    if not all_files:
        print("No CSV files found.")
        return
        
    print(f"Found {len(all_files)} files. Range: {all_files[0][0]} ~ {all_files[-1][0]}")

    # 定義各類別的檔案列表與順序
    # 民事、刑事：從 2015 年 (201501) 開始往後 (由舊到新)
    start_date_2015 = 201501
    files_civil_criminal = [f for d, f in all_files if d >= start_date_2015]
    
    # 行政：從最後面開始往前抽 (由新到舊)
    files_admin = [f for d, f in reversed(all_files)]
    
    print(f"Civil/Criminal: Starting from 201501 ({len(files_civil_criminal)} files)")
    print(f"Administrative: Reverse order from {all_files[-1][0]} ({len(files_admin)} files)")

    # 分類別收集器
    collectors = {
        '民事': [],
        '刑事': [],
        '行政': []
    }
    
    counts = {
        '民事': 0,
        '刑事': 0,
        '行政': 0
    }
    
    # 執行收集：民事與刑事
    print("Collecting Civil and Criminal data...")
    for filename in files_civil_criminal:
        if counts['民事'] >= SAMPLE_SIZE and counts['刑事'] >= SAMPLE_SIZE:
             break
             
        filepath = os.path.join(INPUT_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                rows = list(reader) 
                
                for row in rows:
                    cat = row.get('category')
                    if cat in ['民事', '刑事']:
                        if counts[cat] < SAMPLE_SIZE:
                            row['source_file'] = filename
                            collectors[cat].append(row)
                            counts[cat] += 1
                            if counts[cat] == SAMPLE_SIZE:
                                print(f"[{cat}] Reached {SAMPLE_SIZE} (Last file: {filename})")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # 執行收集：行政
    print("Collecting Administrative data...")
    for filename in files_admin:
        if counts['行政'] >= SAMPLE_SIZE:
            break
            
        filepath = os.path.join(INPUT_DIR, filename)
        try:
            with open(filepath, 'r', encoding='utf-8-sig') as f:
                reader = csv.DictReader(f)
                rows = list(reader)
                
                for row in rows:
                    cat = row.get('category')
                    if cat == '行政':
                        if counts['行政'] < SAMPLE_SIZE:
                            row['source_file'] = filename
                            collectors['行政'].append(row)
                            counts['行政'] += 1
                            if counts['行政'] == SAMPLE_SIZE:
                                print(f"[行政] Reached {SAMPLE_SIZE} (Last file: {filename})")
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    # 3. 寫入結果
    print("\nWriting results...")
    
    for cat, rows in collectors.items():
        if not rows:
            print(f"[{cat}] No data collected")
            continue
            
        output_filename = f"{cat}_continuous_{len(rows)}.csv"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        rows_to_write = rows[:SAMPLE_SIZE]
        
        fieldnames = ['檔名', 'JTITLE', 'category', 'source_file']
        
        try:
            with open(output_path, 'w', newline='', encoding='utf-8-sig') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for row in rows_to_write:
                    clean_row = {k: row.get(k, '') for k in fieldnames}
                    writer.writerow(clean_row)
            print(f"[{cat}] Written to {output_filename} ({len(rows_to_write)} rows)")
            
        except Exception as e:
            print(f"Error writing {cat}: {e}")

if __name__ == "__main__":
    sample_continuous_cases()
