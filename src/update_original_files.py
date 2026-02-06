import os
import csv
import glob

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Path where scraper results are stored
    SERVER_DATA_DIR = os.path.join(BASE_DIR, 'data', 'server_batches')
    
    # Paths where original files to be updated are located
    ORIGINAL_DIRS = [
        os.path.join(BASE_DIR, 'data', 'categorized_case'),
        os.path.join(BASE_DIR, 'data', 'sampled_time_series')
    ]
    
    print("=== Update Original Files from Scraper Data ===")
    
    # 1. Load Scraper Data
    print("Loading scraper results...")
    updates = {} # Map: filename -> score
    cleaned_count = 0
    
    if not os.path.exists(SERVER_DATA_DIR):
        print(f"Scraper data dir {SERVER_DATA_DIR} does not exist.")
        return
        
    batch_files = glob.glob(os.path.join(SERVER_DATA_DIR, 'batch_*.csv'))
    for fpath in batch_files:
        rows_to_keep = [] 
        file_changed = False
        
        with open(fpath, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames if reader.fieldnames else []
            rows = list(reader)
            
        for row in rows:
            filename = row.get('檔名')
            score = row.get('disputability')
            
            if filename:
                # Update logic: '0' -> '' (empty string)
                if score and score.strip() == '0':
                    final_score = ''
                else:
                    final_score = score.strip() if score else ''
                
                # Update in memory for file write back
                if score and score.strip() == '0':
                    row['disputability'] = ''
                    file_changed = True
                    cleaned_count += 1
                
                updates[filename] = row.get('disputability', '')
            
            rows_to_keep.append(row)

        # Write updates back to batch file
        if file_changed:
            with open(fpath, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows_to_keep)
                
    print(f"Loaded {len(updates)} updates. Cleaned {cleaned_count} zero values.")

    # 2. Update Original Files
    print("Updating original files...")
    files_updated = 0
    
    for folder in ORIGINAL_DIRS:
        if not os.path.exists(folder):
            continue
            
        csv_files = glob.glob(os.path.join(folder, '*.csv'))
        
        for fpath in csv_files:
            rows = []
            fieldnames = []
            file_modified = False
            
            with open(fpath, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames if reader.fieldnames else []
                
                # Normalize '檔名' key if BOM exists
                filename_key = '檔名'
                if fieldnames and fieldnames[0].endswith('檔名'):
                     filename_key = fieldnames[0]

                for row in reader:
                    filename = row.get(filename_key)
                    if filename and filename in updates:
                        new_val = updates[filename]
                        current_val = row.get('disputability')
                        if current_val is None: current_val = ''
                        current_val = current_val.strip()
                        
                        if current_val != new_val:
                            row['disputability'] = new_val
                            file_modified = True
                    
                    if 'disputability' not in row:
                        row['disputability'] = ''
                        
                    rows.append(row)
            
            if 'disputability' not in fieldnames:
                fieldnames.append('disputability')
                file_modified = True

            if file_modified:
                with open(fpath, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                files_updated += 1
                
    print(f"Original files updated: {files_updated}")

if __name__ == "__main__":
    main()
