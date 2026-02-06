import pandas as pd
import os
import glob
import re
import numpy as np

def process_datasets():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DIR = os.path.join(BASE_DIR, 'data', 'datasets_csv')
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    tasks = [
        {
            'info': 'Categorized Cases',
            'src_dir': os.path.join(BASE_DIR, 'data', 'categorized_case'),
            'sample_size': 20000
        },
        {
            'info': 'Sampled Time Series',
            'src_dir': os.path.join(BASE_DIR, 'data', 'sampled_time_series'),
            'sample_size': 100000
        }
    ]

    for task in tasks:
        src = task['src_dir']
        target_n = task['sample_size']
        print(f"Processing {task['info']} from {src}...")
        
        if not os.path.exists(src):
            print(f"Source directory not found: {src}")
            continue

        files = glob.glob(os.path.join(src, '*.csv'))
        files.sort()
        
        for fpath in files:
            fname = os.path.basename(fpath)
            try:
                df = pd.read_csv(fpath, low_memory=False)
                
                if 'disputability' not in df.columns:
                    print(f"Skipping {fname}: No 'disputability' column.")
                    continue
                
                # Convert to numeric
                df['disputability_num'] = pd.to_numeric(df['disputability'], errors='coerce')
                
                # Filter: remove NaN and -1
                condition = (df['disputability_num'].notna()) & (df['disputability_num'] != -1)
                valid_df = df[condition].copy()
                
                valid_df['disputability'] = valid_df['disputability_num'].astype(int)
                valid_df.drop(columns=['disputability_num'], inplace=True)
                
                total_valid = len(valid_df)
                if total_valid < target_n:
                    if total_valid > 0:
                        print(f"Warning: {fname} has {total_valid} valid rows (target {target_n}). Taking all.")
                    else:
                        print(f"Warning: {fname} has 0 valid rows. Skipping.")
                        continue
                    sampled_df = valid_df
                else:
                    sampled_df = valid_df.sample(n=target_n, random_state=42)
                
                # Clean Filename (remove count suffix like _30000)
                out_fname = re.sub(r'_\d+\.csv$', '.csv', fname)
                out_path = os.path.join(OUTPUT_DIR, out_fname)
                
                sampled_df.to_csv(out_path, index=False)
                print(f"Saved {out_fname}: {len(sampled_df)} rows")
                
            except Exception as e:
                print(f"Error processing {fname}: {e}")

if __name__ == "__main__":
    process_datasets()
