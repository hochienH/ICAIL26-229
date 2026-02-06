import os
import csv
import glob

def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    TARGET_DIRS = [
        os.path.join(BASE_DIR, 'data', 'categorized_case'),
        os.path.join(BASE_DIR, 'data', 'sampled_time_series')
    ]

    print("=== Add 'disputability' Column Tool ===")
    
    files_modified = 0
    total_files = 0

    for folder in TARGET_DIRS:
        if not os.path.exists(folder):
            print(f"Warning: Directory {folder} does not exist. Skipping.")
            continue
            
        csv_files = glob.glob(os.path.join(folder, '*.csv'))
        print(f"Processing folder: {os.path.basename(folder)} ({len(csv_files)} files)")
        
        for fpath in csv_files:
            total_files += 1
            rows = []
            fieldnames = []
            needs_update = False
            
            # Read
            with open(fpath, 'r', encoding='utf-8', newline='') as f:
                reader = csv.DictReader(f)
                fieldnames = reader.fieldnames if reader.fieldnames else []
                
                if 'disputability' not in fieldnames:
                    # print(f"Adding column to: {os.path.basename(fpath)}")
                    fieldnames.append('disputability')
                    needs_update = True
                
                rows = list(reader)

            # Write back ONLY if needed
            if needs_update:
                with open(fpath, 'w', encoding='utf-8', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerows(rows)
                files_modified += 1
            # else:
                # print(f"Skipping {os.path.basename(fpath)} (Already has column)")

    print(f"\nSummary: Checked {total_files} files. Added column to {files_modified} files.")

if __name__ == "__main__":
    main()
