import pandas as pd
import os
import glob
from tqdm import tqdm
import rarfile
import json
import shutil
import re

# 設定路徑
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASETS_DIR = os.path.join(PROJECT_ROOT, "data/datasets_csv")
DATASETS_MONTHLY_DIR = os.path.join(PROJECT_ROOT, "data/datasets_monthly")
JUDGEMENTS_DIR = os.path.join(PROJECT_ROOT, "data/judgements")
OPENDATA_DIR = os.path.join(PROJECT_ROOT, "data/raw_rar")

def setup_directories():
    """建立必要的資料夾"""
    os.makedirs(DATASETS_MONTHLY_DIR, exist_ok=True)
    os.makedirs(JUDGEMENTS_DIR, exist_ok=True)
    print(f"Directories created: {DATASETS_MONTHLY_DIR}, {JUDGEMENTS_DIR}")

def process_datasets_to_monthly():
    """
    1. 讀取 data/datasets 下的所有 .csv
    2. concat 起來
    3. 根據 source_file 分群
    4. 存入 data/datasets_monthly
    """
    # Check if we already have monthly files to avoid re-processing
    existing_files = glob.glob(os.path.join(DATASETS_MONTHLY_DIR, "*.csv"))
    if existing_files and len(existing_files) > 10: # Simple heuristic
        print(f"Found {len(existing_files)} existing monthly datasets in {DATASETS_MONTHLY_DIR}. Skipping splitting process.")
        monthly_files_map = {}
        for f in existing_files:
            run_name = os.path.splitext(os.path.basename(f))[0]
            monthly_files_map[run_name] = f
        return monthly_files_map

    print("Reading and aggregating datasets...")
    all_files = glob.glob(os.path.join(DATASETS_DIR, "*.csv"))
    
    # 讀取並合併所有 CSV
    df_list = []
    for f in tqdm(all_files, desc="Loading CSVs"):
        try:
            # 假設 csv 格式與民事_分割共有物.csv 相同，有 JID, source_file 等欄位
            df = pd.read_csv(f)
            df_list.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")
            
    if not df_list:
        print("No csv files found.")
        return

    full_df = pd.concat(df_list, ignore_index=True)
    
    # 確保 source_file 欄位存在 (從範例看應該有)
    if 'source_file' not in full_df.columns:
        # 如果 datasets 沒有 source_file，可能要從 JID 推斷或檢查資料來源
        # 這裡假設 user 描述的 datasets 裡有 source_file
        print("Error: 'source_file' column not found in datasets.")
        return

    # 根據 source_file 分組並儲存
    # source_file 格式可能是 path/to/200001.csv 或只是 200001
    # 這裡我們只取檔名部分作為 key
    
    print("Splitting into monthly datasets...")
    grouped = full_df.groupby('source_file')
    
    monthly_files_map = {} # source_filename -> output_path

    for source_file, group in tqdm(grouped, desc="Saving monthly"):
        # 清理 source_file 名稱以作為檔名，假設 source_file 類似 "200001.csv" 或 "200001"
        # 這裡取單純的檔名部分 (去掉路徑，保留副檔名如果有的話，或者直接用)
        base_name = os.path.basename(str(source_file))
        # 移除 .csv 如果有的話，拿到像 "200001" 這樣的字串
        run_name = os.path.splitext(base_name)[0]
        
        output_path = os.path.join(DATASETS_MONTHLY_DIR, f"{run_name}.csv")
        group.to_csv(output_path, index=False)
        monthly_files_map[run_name] = output_path
        
    return monthly_files_map

def get_last_processed_month(judgements_dir):
    try:
        if not os.path.exists(judgements_dir): return None
        with os.scandir(judgements_dir) as it:
            # Only look at json files to avoid .DS_Store or other noise
            files = (e for e in it if e.is_file() and e.name.lower().endswith('.json'))
            try: latest = max(files, key=lambda e: e.stat().st_mtime)
            except ValueError: return None
        match = re.search(r'(19|20)\d{2}(0[1-9]|1[0-2])', latest.name)
        return match.group(0)[:6] if match else None
    except Exception: return None

def extract_judgements_from_rar(monthly_files_map):
    """
    根據 monthly datasets 的檔名 (e.g., 200001) 去找 OPENDATA_DIR 下的對應 rar (200001.rar)
    遍歷 rar，如果檔名在 csv 裡，就解壓出來
    """
    print("Extracting judgements from RARs...")
    
    start_month = get_last_processed_month(JUDGEMENTS_DIR)
    if start_month:
        print(f"Found last processed month: {start_month}. Resuming from {start_month} (inclusive) to ensure completion.")

    sorted_keys = sorted(monthly_files_map.keys())
    skipping = True if start_month else False
    
    # Check if rarfile is available (system dependency: unrar)
    # rarfile 需要系統安裝 unrar / rar 命令行工具
    
    for run_name in tqdm(sorted_keys, desc="Processing RARs"):
        if skipping:
            if run_name == start_month:
                skipping = False
            else:
                continue
        monthly_csv_path = monthly_files_map[run_name]
        rar_path = os.path.join(OPENDATA_DIR, f"{run_name}.rar")
        
        if not os.path.exists(rar_path):
            print(f"RAR file not found: {rar_path}, skipping.")
            continue
            
        # 讀取該月份需要抽取的 JID 列表
        # 我們假設 CSV 裡面的 'JID' 欄位對應到 json 檔名 (e.g. JID='A,123' -> A,123.json)
        # 或者 CSV 裡已經有完整 json 檔名?
        # 通常 JID = "TPSV,89,台上,123,20000101", 實際檔名可能是 "TPSV,89,台上,123,20000101.json"
        # 這裡需要根據實際資料格式確認。我們假設目標檔名就是 JID + ".json"
            
        try:
            df = pd.read_csv(monthly_csv_path)
            # 建立一個 target set，方便快速查找
            # 注意：這裡假設 JID 欄位就是檔名 (不含 .json)。如果 CSV 裡有 filename 欄位更好。
            # 從先前的 context 看，民事_分割共有物.csv 有 JID
            
            target_filenames = set()
            
            # Check for '檔名' column (primary) or 'JID'
            if '檔名' in df.columns:
                # '檔名' column usually contains full filename like "TCDV,103,訴,2396,20141212,1.json"
                target_filenames = set(df['檔名'].astype(str))
            elif 'JID' in df.columns:
                # 'JID' column might need .json extension
                target_filenames = set(df['JID'].astype(str) + ".json")
            else:
                 print(f"Warning: No '檔名' or 'JID' column in {monthly_csv_path}, columns: {df.columns.tolist()}")
                 continue
                 
            # print(f"Targeting {len(target_filenames)} files from {monthly_csv_path}")
                 
            # 開啟 RAR 檔
            with rarfile.RarFile(rar_path) as rf:
                # 取得 rar 內所有檔案列表
                # rf.namelist() 會有所有檔案路徑
                for f in rf.infolist():
                    # 檢查是否為我們要的檔案
                    # f.filename 可能是 "200001/TPSV,89,台上,123,20000101.json"
                    fname = os.path.basename(f.filename)
                    
                    if fname in target_filenames:
                        # 解壓縮到 JUDGEMENTS_DIR
                        # rf.extract(f, JUDGEMENTS_DIR) 會保留目錄結構嗎？
                        # 我們希望能扁平化放到 data/judgements
                        
                        # 使用 read 读取内容并写入新文件，实现扁平化
                        try:
                            content = rf.read(f)
                            dest_path = os.path.join(JUDGEMENTS_DIR, fname)
                            with open(dest_path, 'wb') as out_f:
                                out_f.write(content)
                        except Exception as e:
                             print(f"Failed to extract {fname} from {rar_path}: {e}")

        except rarfile.Error as e:
            print(f"RAR Error processing {rar_path}: {e}")
        except Exception as e:
            print(f"Error processing {monthly_csv_path}: {e}")

def main():
    # 確保系統有 unrar (rarfile 需要)
    # macos: brew install rar / unrar
    # 如果 user 環境沒有，程式會報錯。
    
    setup_directories()
    monthly_map = process_datasets_to_monthly()
    if monthly_map:
        extract_judgements_from_rar(monthly_map)
    print("Done.")

if __name__ == "__main__":
    main()
