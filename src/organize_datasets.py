import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import defaultdict
import sys

# Configuration
# Detect environment (Local vs Server structure)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Typically BASE_DIR is .../projects/Disputability-ICAIL if script is in program/

# Default Paths (Adjustable via args or environment)
DATASETS_CSV_DIR = os.path.join(BASE_DIR, "data", "datasets_csv")
EMBEDDINGS_DIR = os.path.join(BASE_DIR, "data", "embeddings_output")
OUTPUT_PARQUET_DIR = os.path.join(BASE_DIR, "data", "datasets_parquet")

def load_dataset_mappings(csv_dir):
    """
    Reads all CSVs in csv_dir and returns:
    1. file_to_datasets: filename -> list of dataset_names
    2. file_to_metadata: filename -> {disputability, category}
    """
    print(f"Loading dataset definitions from {csv_dir}...")
    csv_files = glob.glob(os.path.join(csv_dir, "*.csv"))
    
    file_to_datasets = defaultdict(list)
    file_to_metadata = {}
    dataset_stats = defaultdict(int)
    
    for fpath in tqdm(csv_files, desc="Reading CSVs"):
        dataset_name = os.path.splitext(os.path.basename(fpath))[0]
        try:
            df = pd.read_csv(fpath)
            # Find filename column
            col_name = '檔名' if '檔名' in df.columns else 'JID' # Fallback
            
            if col_name not in df.columns:
                print(f"Warning: Could not find filename column in {dataset_name}.csv. Columns: {df.columns}")
                continue
            
            # Find disputability column
            disp_col = 'disputability'
            if disp_col not in df.columns:
                 print(f"Warning: Could not find 'disputability' column in {dataset_name}.csv.")
                 continue

            # Find category column
            cat_col = 'category'
            if cat_col not in df.columns:
                 print(f"Warning: Could not find 'category' column in {dataset_name}.csv.")
                 continue

            # Find JTITLE column
            jtitle_col = 'JTITLE'
            if jtitle_col not in df.columns:
                 print(f"Warning: Could not find 'JTITLE' column in {dataset_name}.csv.")
                 # It might be optional, but for now let's assume it exists or verify
                 # If missing, we might want to default to empty string, but prompt implies it's there.
                 # Let's fallback to empty string if missing to be safe
                 jtitles = [''] * len(df)
            else:
                 jtitles = df[jtitle_col].astype(str).values

            # Iterate through rows
            filenames = df[col_name].astype(str).values
            disputabilities = df[disp_col].values
            categories = df[cat_col].values
            
            for fname, disp, cat, jtitle in zip(filenames, disputabilities, categories, jtitles):
                if not fname.endswith('.json'):
                    fname += '.json'
                
                file_to_datasets[fname].append(dataset_name)
                # Store metadata
                file_to_metadata[fname] = {
                    'disputability': disp,
                    'category': cat,
                    'jtitle': jtitle
                }
                dataset_stats[dataset_name] += 1
                
        except Exception as e:
            print(f"Error reading {fpath}: {e}")
            
    print(f"Mapped {len(file_to_datasets)} unique files across {len(dataset_stats)} datasets.")
    return file_to_datasets, file_to_metadata, list(dataset_stats.keys())

def organize_embeddings():
    # 1. Setup
    os.makedirs(OUTPUT_PARQUET_DIR, exist_ok=True)
    
    file_to_datasets, file_to_metadata, dataset_names = load_dataset_mappings(DATASETS_CSV_DIR)
    
    # Check if we have embeddings
    emb_files = glob.glob(os.path.join(EMBEDDINGS_DIR, "*.parquet"))
    if not emb_files:
        print(f"No embedding files found in {EMBEDDINGS_DIR}")
        return

    print(f"Found {len(emb_files)} embedding batch files.")
    
    # 2. Initialize Buffers for each dataset
    dataset_buffers = defaultdict(list)
    
    # 3. Iterate and Distribute
    processing_stats = {"processed_embeddings": 0, "matched": 0, "unmatched": 0}
    
    for emb_file in tqdm(emb_files, desc="Processing Batches"):
        try:
            df = pd.read_parquet(emb_file)
            
            # Convert to records for faster iteration
            records = df.to_dict('records')
            
            for row in records:
                fname = row['filename']
                processing_stats['processed_embeddings'] += 1
                
                if fname in file_to_datasets:
                    target_datasets = file_to_datasets[fname]
                    
                    # Get shared attributes
                    metadata = file_to_metadata.get(fname)
                    
                    processing_stats['matched'] += 1
                    
                    for ds_name in target_datasets:
                        # Create a copy for this specific dataset entry
                        row_copy = row.copy()
                        if metadata:
                            row_copy['disputability'] = metadata['disputability']
                            row_copy['category'] = metadata['category']
                            row_copy['jtitle'] = metadata.get('jtitle', '')
                        
                        dataset_buffers[ds_name].append(row_copy)
                else:
                    processing_stats['unmatched'] += 1
                    
        except Exception as e:
            print(f"Error reading parquet {emb_file}: {e}")

    print("\nEncoding statistics:")
    print(f"Total embeddings processed: {processing_stats['processed_embeddings']}")
    print(f"Matched to datasets: {processing_stats['matched']}")
    print(f"Unmatched (not in any CSV): {processing_stats['unmatched']}")

    # 4. Write Outputs
    print("\nWriting organized parquet files...")
    for ds_name, rows in tqdm(dataset_buffers.items(), desc="Saving Datasets"):
        if not rows:
            continue
            
        out_path = os.path.join(OUTPUT_PARQUET_DIR, f"{ds_name}.parquet")
        try:
            out_df = pd.DataFrame(rows)
            # Optimize schema if needed, but default is usually fine
            out_df.to_parquet(out_path, index=False)
        except Exception as e:
            print(f"Failed to write {ds_name}.parquet: {e}")
            
    # Check for empty datasets (those in CSV but no embeddings found)
    processed_datasets = set(dataset_buffers.keys())
    all_datasets = set(dataset_names)
    missing = all_datasets - processed_datasets
    if missing:
        print(f"Warning: No embeddings found for datasets: {missing}")
    else:
        print("All datasets have been populated.")

if __name__ == "__main__":
    # If file path provided as argument, we can use it, but default logic is better
    # to be environment specific in organizing tasks
    
    # Check if we are running locally with downloaded data
    # Path: server_deploy/data/embeddings_output
    local_downloaded_path = os.path.join(BASE_DIR, "server_deploy", "data", "embeddings_output")
    
    if os.path.exists(local_downloaded_path) and os.listdir(local_downloaded_path):
        print(f"Detected downloaded embeddings at: {local_downloaded_path}")
        EMBEDDINGS_DIR = local_downloaded_path
    
    organize_embeddings()
