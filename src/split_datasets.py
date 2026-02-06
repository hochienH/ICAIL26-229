import os
import pandas as pd
from sklearn.model_selection import train_test_split

def split_dataset(file_path, output_dir):
    filename = os.path.basename(file_path)
    # Remove extension and possible '_continuous' suffix for cleaner output names,
    # or just append split type.
    # Let's keep it simple: "刑事_continuous.parquet" -> "刑事_train.parquet", etc.
    # Actually, simpler names like "刑事_train.parquet" are usually preferred.
    
    base_name = os.path.splitext(filename)[0]
    category_prefix = base_name.replace("_continuous", "")
    
    print(f"Processing {filename} (Category: {category_prefix})...")
    
    try:
        df = pd.read_parquet(file_path)
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return

    total_len = len(df)
    print(f"  Total records: {total_len}")

    # Create Stratify Key to handle rare classes
    # If a jtitle has fewer than 5 samples, treat it as a 'rare' group to allow stratification on the rest
    if 'jtitle' in df.columns:
        counts = df['jtitle'].value_counts()
        rare_titles = counts[counts < 5].index
        df['stratify_key'] = df['jtitle'].apply(lambda x: 'RARE_GROUP' if x in rare_titles else x)
        
        # Check if RARE_GROUP itself is too small (e.g. only 1 rare item total)
        # If stratify_key has any class < 2 members, stratification fails.
        # We group those into RARE_GROUP as well (effectively recursive, but here simple check)
        s_counts = df['stratify_key'].value_counts()
        if 'RARE_GROUP' in s_counts and s_counts['RARE_GROUP'] < 5:
             # Even the combined rare group is too small? 
             # Just map everything small to 'RARE_GROUP' was the step above.
             # If RARE_GROUP < 5, it means total rare items are few.
             # We might not be able to stratify RARE_GROUP if it has 1 item.
             pass
    else:
        df['stratify_key'] = "ALL"


    # Split: 80% Train, 20% Temp
    # Try Stratified Split first
    try:
        print("  Attempting stratified split (by jtitle)...")
        train_df, temp_df = train_test_split(
            df, 
            test_size=0.2, 
            random_state=42, 
            shuffle=True, 
            stratify=df['stratify_key']
        )
    except ValueError as e:
        print(f"  Warning: Stratification failed ({e}). Falling back to random split.")
        train_df, temp_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    
    # Split Temp: 50% Val (10% total), 50% Test (10% total)
    # Re-evaluate stratification for the second split
    try:
        # Check counts in temp_df to see if we can still stratify
        temp_counts = temp_df['stratify_key'].value_counts()
        rare_2 = temp_counts[temp_counts < 2].index
        
        if len(rare_2) > 0:
            # Regroup strictly for this split
            temp_df = temp_df.copy()
            temp_df['stratify_key_2'] = temp_df['stratify_key'].apply(lambda x: 'RARE_2' if x in rare_2 else x)
            strat_col = temp_df['stratify_key_2']
        else:
            strat_col = temp_df['stratify_key']

        val_df, test_df = train_test_split(
            temp_df, 
            test_size=0.5, 
            random_state=42, 
            shuffle=True, 
            stratify=strat_col
        )
    except Exception as e:
        print(f"  Warning: Second stratification failed ({e}). Random split used for Val/Test.")
        val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=42, shuffle=True)
    
    # Cleanup helper columns
    for d in [train_df, val_df, test_df, df]:
        if 'stratify_key' in d.columns: del d['stratify_key']
        if 'stratify_key_2' in d.columns: del d['stratify_key_2']
    
    # Verify sizes
    print(f"  Train: {len(train_df)} ({len(train_df)/total_len:.1%})")
    print(f"  Val:   {len(val_df)} ({len(val_df)/total_len:.1%})")
    print(f"  Test:  {len(test_df)} ({len(test_df)/total_len:.1%})")
    
    # Save
    train_path = os.path.join(output_dir, f"{category_prefix}_train.parquet")
    val_path = os.path.join(output_dir, f"{category_prefix}_val.parquet")
    test_path = os.path.join(output_dir, f"{category_prefix}_test.parquet")
    
    train_df.to_parquet(train_path)
    val_df.to_parquet(val_path)
    test_df.to_parquet(test_path)
    
    print(f"  Saved to:\n    {train_path}\n    {val_path}\n    {test_path}\n")

def main():
    # Use path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # Source directory should be 'data/datasets_parquet' where the large continuous files are
    data_dir = os.path.join(project_root, "data", "datasets_parquet")
    # Output directory
    output_dir = os.path.join(project_root, "data", "datasets_split")
    os.makedirs(output_dir, exist_ok=True)
    
    # Files to process
    target_files = [
        "刑事_continuous.parquet",
        "民事_continuous.parquet",
        "行政_continuous.parquet"
    ]
    
    for fname in target_files:
        fpath = os.path.join(data_dir, fname)
        if os.path.exists(fpath):
            split_dataset(fpath, output_dir)
        else:
            print(f"Warning: File not found: {fpath}")

if __name__ == "__main__":
    main()
