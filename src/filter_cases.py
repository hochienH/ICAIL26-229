import os
import json
import csv
import rarfile
from pathlib import Path
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

# 設定 unrar 路徑 (May need adjustment depending on OS)
# rarfile.UNRAR_TOOL = "/opt/homebrew/bin/unrar" 

# Dynamic Base Directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'first_instance_case')

def get_file_category(dirname, filename):
    """
    根據目錄名稱和檔名判斷案件類別。
    """
    # 共同排除條件
    if "最高" in dirname:
        return None
        
    # 新增排除規則：檔名排除 '更', '再'
    if "更" in filename or "再" in filename:
        return None

    # 行政訴訟檢查 (注意：這必須在檢查 "高等" 排除之前，因為行政訴訟允許 "高等")
    if "行政" in dirname:
        if "上" not in filename:
            return "行政"
            
    # 民事與刑事的共同排除條件
    if "高等" in dirname:
        return None
    
    # 民事檢查
    if "民事" in dirname:
        if "上" not in filename:
            return "民事"
            
    # 刑事檢查
    if "刑事" in dirname:
        if "上" not in filename and "附民" not in filename:
            return "刑事"
            
    return None

def is_valid_jfull(jfull):
    """
    檢查 JFULL 屬性是否符合條件：
    1. 頭 30 字裡有 "判決"
    """
    if not jfull:
        return False
    # 檢查前 30 個字元
    return "判決" in jfull[:30]

def process_rar_file(rar_path):
    """
    處理單個 RAR 檔案並生成對應的 CSV
    """
    rar_path = Path(rar_path)
    output_filename = rar_path.with_suffix('.csv').name
    output_file = os.path.join(OUTPUT_DIR, output_filename)
    print(f"正在處理: {rar_path.name}")
    
    results = []
    count_processed = 0
    count_matched = 0
    
    try:
        with rarfile.RarFile(rar_path) as rf:
            for infolist in rf.infolist():
                if infolist.isdir():
                    continue
                if not infolist.filename.endswith('.json'):
                    continue
                    
                # 解析路徑以獲取上層目錄名稱
                file_path = infolist.filename
                parts = file_path.split('/')
                if len(parts) > 1:
                    dirname = parts[-2]
                else:
                    dirname = "" # 根目錄下的檔案
                
                filename = parts[-1]
                
                # 判斷類別
                category = get_file_category(dirname, filename)
                if not category:
                    continue
                
                count_processed += 1
                if count_processed % 1000 == 0:
                    print(f"  已掃描 {count_processed} 個潛在檔案...", end='\r')
                
                try:
                    # 直接從 RAR 讀取，不解壓縮到磁碟
                    with rf.open(infolist) as f:
                        content = f.read().decode('utf-8')
                        data = json.loads(content)
                        
                        jfull = data.get('JFULL', '')
                        jtitle = data.get('JTITLE', '')
                        
                        # 檢查 JFULL 內容條件
                        if is_valid_jfull(jfull):
                            results.append({
                                'filename': filename,
                                'jtitle': jtitle,
                                'category': category
                            })
                            count_matched += 1
                            
                except Exception as e:
                    print(f"\n  讀取RAR內檔案 {filename} 時發生錯誤: {e}")

        print(f"\n  處理完成。共檢查 {count_processed} 個相關檔案，找到 {count_matched} 個符合條件。")
        
        # 寫入 CSV
        if results:
            with open(output_file, 'w', newline='', encoding='utf-8-sig') as csvfile:
                fieldnames = ['檔名', 'JTITLE', 'category']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                writer.writeheader()
                for row in results:
                    writer.writerow({
                        '檔名': row['filename'],
                        'JTITLE': row['jtitle'],
                        'category': row['category']
                    })
            print(f"  已產生報告: {output_file}")
            return True
        else:
            print(f"  沒有符合條件的檔案，未產生報告。")
            return False

    except rarfile.BadRarFile:
        print(f"錯誤: {rar_path.name} 是損壞的 RAR 檔案")
        return False
    except Exception as e:
        print(f"處理 {rar_path.name} 時發生未預期的錯誤: {e}")
        return False

def main():
    # Input Directory (User should place raw .rar files here)
    source_dir = os.path.join(BASE_DIR, 'data', 'raw_rar')
    source_path = Path(source_dir)
    
    if not source_path.exists():
        print(f"Warning: Source directory {source_path} does not exist.")
        print("Please create it and place the monthly .rar files (e.g., 202001.rar) from Judicial Yuan Open Data there.")
        return

    rar_files = sorted(list(source_path.glob("*.rar")))
    
    if not rar_files:
        print(f"No .rar files found in {source_path}")
        return
        
    print(f"Found {len(rar_files)} RAR files.")
    
    # 確保輸出目錄存在
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
    
    # 準備此批次要處理的檔案
    files_to_process = []
    
    for rar_file in rar_files:
        # 檢查輸出檔案是否已存在
        output_filename = rar_file.with_suffix('.csv').name
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        if os.path.exists(output_path):
            print(f"Skipping: {rar_file.name} (CSV already exists)")
            continue
        
        files_to_process.append(rar_file)
    
    if not files_to_process:
        print("All files processed.")
        return

    print(f"Processing {len(files_to_process)} files using 8 parallel processes...")
    start_time = time.time()
    
    with ProcessPoolExecutor(max_workers=8) as executor:
        future_to_file = {executor.submit(process_rar_file, rar): rar for rar in files_to_process}
        
        for future in as_completed(future_to_file):
            rar = future_to_file[future]
            try:
                future.result()
            except Exception as e:
                print(f"Exception processing {rar.name}: {e}")
                
    end_time = time.time()
    print(f"Completed. Total time: {end_time - start_time:.2f} seconds")

if __name__ == "__main__":
    main()
