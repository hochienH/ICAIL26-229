import pandas as pd
import numpy as np
import os
import sys
import torch
import warnings
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, f1_score
from collections import defaultdict
import joblib

# Add local path to import existing modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from train_shn import SemanticHurdleNetwork, ZTGPLoss, transform_target, THRESHOLD_MAP, compute_centroids, train_shn, predict_shn, evaluate_metrics, get_available_gpus
    from optimized_clustering import FaissKMeans, HAS_FAISS, SemanticClusteringManager, JudgmentDataset
except ImportError as e:
    print(f"Error importing modules: {e}")
    sys.exit(1)

warnings.filterwarnings('ignore')

# Helper for Logging
class Logger(object):
    def __init__(self, filename):
        self.terminal = sys.stdout
        self.log = open(filename, "w", buffering=1)

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

# Helper Wrapper for Optimized Clustering
class SimpleDatasetWrapper:
    """Matches the interface expected by SemanticClusteringManager"""
    def __init__(self, df):
        self.df = df
        # Create statistics immediately
        self.df['coa_label'] = self.df['jtitle'].astype(str)
        self.df['disputability'] = pd.to_numeric(self.df['disputability'], errors='coerce').fillna(0)
        
        # Build vector matrix
        if 'dense_vec' in df.columns:
             self.vectors = np.stack(df['dense_vec'].values)
        else:
             raise ValueError("dense_vec column missing")
             
        # Mapping
        self.coa_unique = self.df['coa_label'].unique()
        self.coa_to_id = {label: idx for idx, label in enumerate(self.coa_unique)}
        self.df['coa_id'] = self.df['coa_label'].map(self.coa_to_id)

    def get_coa_statistics(self):
        counts = self.df['coa_label'].value_counts()
        valid_coas = counts[counts >= 5].index
        stats = self.df[self.df['coa_label'].isin(valid_coas)].groupby('coa_label')['disputability'].mean().to_dict()
        return stats

# ==========================================
# Configuration
# ==========================================
DATA_DIR_PARQUET = "data/datasets_parquet"
DATA_DIR_SPLIT = "data/datasets_split"
RESULTS_DIR = "ablation_results"
os.makedirs(RESULTS_DIR, exist_ok=True)

CATEGORIZED_FILES = {
    '刑事': [
        '刑事_傷害.parquet', '刑事_公共危險.parquet', 
        '刑事_毒品危害防制條例.parquet', '刑事_竊盜_詐欺.parquet'
    ],
    '民事': [
        '民事_分割共有物.parquet', '民事_損害賠償.parquet', 
        '民事_清償借款_清償債務.parquet', '民事_離婚.parquet'
    ],
    '行政': [
        '行政_交通裁決.parquet', '行政_稅.parquet'
    ]
}

# ==========================================
# Helper Class for SemanticClusteringManager
# ==========================================
class SimpleDatasetWrapper:
    """
    Wraps a pandas DataFrame to match the interface expected by SemanticClusteringManager.
    Expects df to have 'dense_vec', 'jtitle', 'disputability'.
    """
    def __init__(self, df):
        self.df = df.reset_index(drop=True)
        
        # Prepare vectors
        print("    [Wrapper] Stack vectors...")
        self.vectors = np.stack(self.df['dense_vec'].values)
        
        # Prepare COA (JTitle) info
        self.df['coa_label'] = self.df['jtitle'].astype(str)
        self.df['disputability'] = pd.to_numeric(self.df['disputability'], errors='coerce').fillna(0)
        
        self.coa_unique = self.df['coa_label'].unique()
        self.coa_to_id = {label: idx for idx, label in enumerate(self.coa_unique)}
        self.df['coa_id'] = self.df['coa_label'].map(self.coa_to_id)
        
        print(f"    [Wrapper] Ready. {len(self.df)} samples, {len(self.coa_unique)} COAs.")

    def get_coa_statistics(self):
        """Re-implementation of JudgmentDataset.get_coa_statistics"""
        counts = self.df['coa_label'].value_counts()
        valid_coas = counts[counts >= 5].index
        stats = self.df[self.df['coa_label'].isin(valid_coas)].groupby('coa_label')['disputability'].mean().to_dict()
        return stats

# ==========================================
# Data Preparation
# ==========================================

def load_and_merge_categorized(category):
    """Load and merge specific categorized files for a given category."""
    files = CATEGORIZED_FILES.get(category, [])
    dfs = []
    print(f"[{category}] Loading categorized files: {files}")
    for f in files:
        path = os.path.join(DATA_DIR_PARQUET, f)
        if os.path.exists(path):
            df = pd.read_parquet(path)
            dfs.append(df)
        else:
            print(f"Warning: File not found {path}")
            
    if not dfs:
        raise ValueError(f"No data found for category {category}")
        
    full_df = pd.concat(dfs, ignore_index=True)
    print(f"[{category}] Merged Data Shape: {full_df.shape}")
    return full_df

def split_stratified(df, stratify_col='jtitle'):
    """Split dataframe 8:1:1 stratify by jtitle, with fallback for rare classes."""
    # Ensure min samples for stratification (need at least 2 for split, safer with 5 for 3-way split)
    v_counts = df[stratify_col].value_counts()
    # If a class has < 5 samples, it's hard to split 8:1:1 while maintaining stratification in all steps
    # We will try best effort stratification.
    
    # Strategy:
    # 1. Separate "Robust Classes" (>5 samples) and "Rare Classes"
    # 2. Stratify split robust classes
    # 3. Randomly split rare classes (or just put them in train to be safe)
    
    robust_classes = v_counts[v_counts >= 5].index
    
    df_robust = df[df[stratify_col].isin(robust_classes)].copy()
    df_rare = df[~df[stratify_col].isin(robust_classes)].copy()
    
    # print(f"  Stratification: {len(df_robust)} robust samples, {len(df_rare)} rare samples.")
    
    # Split Robust: 8:2 (Train : Temp)
    train_robust, temp_robust = train_test_split(
        df_robust, test_size=0.2, stratify=df_robust[stratify_col], random_state=42
    )
    
    # Split Temp Robust: 50:50 (Val : Test)
    # Even with >=5 samples total, after 20% split, we might have 1 sample in temp.
    # So we use try-except fallback for the second split too.
    try:
        val_robust, test_robust = train_test_split(
            temp_robust, test_size=0.5, stratify=temp_robust[stratify_col], random_state=42
        )
    except ValueError:
        # Fallback to random split if stratification fails in the second step
        # print("  Warning: Secondary stratification failed. Falling back to random split for validation.")
        val_robust, test_robust = train_test_split(
            temp_robust, test_size=0.5, random_state=42
        )

    # Handle Rare: Just assign to Train to avoid errors and data loss
    # Or split randomly 8:1:1 if possible, but simplest is 100% train for rare stuff
    # to ensure model sees them at least once? 
    # Let's try to split rare randomly 8:2 then 5:5 to keep distribution similar-ish
    if len(df_rare) > 0:
        if len(df_rare) >= 5:
            train_rare, temp_rare = train_test_split(df_rare, test_size=0.2, random_state=42)
            val_rare, test_rare = train_test_split(temp_rare, test_size=0.5, random_state=42)
        else:
            # Too few to split meaningfuly, put all in train or split arbitrarily
            train_rare = df_rare
            val_rare = pd.DataFrame(columns=df.columns)
            test_rare = pd.DataFrame(columns=df.columns)
            
        train_df = pd.concat([train_robust, train_rare])
        val_df = pd.concat([val_robust, val_rare])
        test_df = pd.concat([test_robust, test_rare])
    else:
        train_df, val_df, test_df = train_robust, val_robust, test_robust
    
    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)

def prepare_datasets(category, dataset_type):
    """
    Prepare Train/Val/Test dataframes.
    
    Args:
        category: '刑事', '民事', '行政'
        dataset_type: 'Categorized' or 'Continuous'
        
    Returns:
        dict: {'train': df, 'val': df, 'test': df}
    """
    datasets = {}
    
    if dataset_type == 'Continuous':
        # Load existing splits
        for split in ['train', 'val', 'test']:
            path = os.path.join(DATA_DIR_SPLIT, f"{category}_{split}.parquet")
            print(f"  Loading {path}...")
            datasets[split] = pd.read_parquet(path)
            
    elif dataset_type == 'Categorized':
        # Merge and Split
        full_df = load_and_merge_categorized(category)
        train, val, test = split_stratified(full_df)
        datasets['train'] = train
        datasets['val'] = val
        datasets['test'] = test
        
    return datasets

# ==========================================
# Clustering Logic
# ==========================================

def run_optimized_clustering_pipeline(datasets, category_name="Unknown"):
    """
    Run the full Semantic Clustering Optimization Pipeline:
    1. Wrap Train Data
    2. Representative Sampling
    3. Grid Search for K (Dual-Objective)
    4. Predict on Train/Val/Test
    """
    print(f"  [Clustering] Starting Optimization Pipeline for {category_name}...")
    
    # 1. Wrap Train Data
    train_df = datasets['train']
    wrapped_dataset = SimpleDatasetWrapper(train_df)
    
    # 2. Initialize Manager & Sample
    manager = SemanticClusteringManager(wrapped_dataset, m_support=100)
    manager.step1_representative_sampling()
    
    n_sampled_coas = len(np.unique(manager.prototype_coa_ids))
    print(f"  [Clustering] Sampled COAs: {n_sampled_coas}")
    
    # 3. Step 2: Grid Search
    # Heuristics from run_parallel_clustering.py
    if n_sampled_coas < 5:
        print("  [Clustering] Way too few COAs. Fallback to simple K=min(20, n_sampled).")
        best_k = max(2, n_sampled_coas) 
        if HAS_FAISS:
             best_model = FaissKMeans(n_clusters=best_k, use_gpu=True)
        else:
             from sklearn.cluster import KMeans
             best_model = KMeans(n_clusters=best_k, n_init=10, random_state=42)
        best_model.fit(wrapped_dataset.vectors)
    else:
        k_min = max(2, int(n_sampled_coas * 0.05))
        k_max = min(n_sampled_coas, int(n_sampled_coas * 0.9)) # Fixed upper bound logic
        if k_max <= k_min: k_max = k_min + 5
        
        # Grid search step fixed to 1 as requested
        step = 1
        
        print(f"  [Clustering] Grid Search K: [{k_min}, {k_max}], Step={step}")
        
        best_model, best_k, _ = manager.step2_dual_objective_grid_search(
            k_min=k_min, k_max=k_max, step=step, lambda_div=0.1
        )
    
    print(f"  [Clustering] Optimal K Found: {best_k}")
    
    # 4. Predict on All Splits using the Best Model
    # Note: best_model might be sklearn KMeans or FaissKMeans depending on what step2 returns
    # step2 returns sklearn KMeans usually. FaissKMeans wrapper mimics predict interface.
    
    # Helper to predict in batches
    def batch_predict(model, X, bsize=5000):
        preds = []
        for i in range(0, len(X), bsize):
            batch = X[i:i+bsize]
            preds.extend(model.predict(batch))
        return np.array(preds)

    # Train (Using already loaded vectors in wrapped_dataset to save stack time?)
    # But datasets['train'] might have been copied. Use fresh vectors just in case.
    print("  [Clustering] Predicting Cluster IDs...")
    datasets['train']['cluster_id'] = batch_predict(best_model, np.stack(datasets['train']['dense_vec'].values))
    datasets['val']['cluster_id'] = batch_predict(best_model, np.stack(datasets['val']['dense_vec'].values))
    datasets['test']['cluster_id'] = batch_predict(best_model, np.stack(datasets['test']['dense_vec'].values))
    
    return datasets, best_k

def run_jtitle_clustering(datasets):
    """
    Map jtitle to cluster_id directly.
    """
    print("  [Clustering] Using JTitle as Cluster ID (No Re-clustering)...")
    
    # Learn mapping from Train
    le = LabelEncoder()
    # We fit on all known jtitles across splits to avoid errors, 
    # but in a strict ML sense we should handle unknowns. 
    # For this ablation, we assume closed set or sufficient coverage.
    all_jtitles = pd.concat([
        datasets['train']['jtitle'], 
        datasets['val']['jtitle'], 
        datasets['test']['jtitle']
    ]).unique()
    
    le.fit(all_jtitles)
    n_clusters = len(le.classes_)
    
    print(f"  [Clustering] Found {n_clusters} unique jtitles.")
    
    for split in ['train', 'val', 'test']:
        datasets[split]['cluster_id'] = le.transform(datasets[split]['jtitle'])
        
    return datasets, n_clusters

def prepare_shn_input(datasets, category):
    """Convert dataframes to the dict format expected by train_shn."""
    formatted_data = {}
    for split, df in datasets.items():
        X_dense = np.stack(df['dense_vec'].values)
        cluster_ids = df['cluster_id'].values.astype(int)
        raw_y = df['disputability'].values.astype(float)
        transformed_y = transform_target(raw_y, category)
        
        formatted_data[split] = {
            'dense': X_dense,
            'cluster': cluster_ids,
            'y': transformed_y,
            'raw_y': raw_y
        }
    return formatted_data

# ==========================================
# Main Experiment Loop
# ==========================================

import multiprocessing
import time

# ==========================================
# Main Experiment Loop (Parallelized)
# ==========================================

def run_category_ablation(category, gpu_id):
    """
    Worker function to run ablation study for a single category on a specific GPU.
    """
    # Setup thread-safe/process-safe logging
    log_file = os.path.join(RESULTS_DIR, f"log_ablation_{category}.txt")
    sys.stdout = Logger(log_file)
    print(f"[{category}] Process started on GPU {gpu_id}. Logging to {log_file}")
    
    results_summary = []
    types = ['Categorized', 'Continuous']
    methods = ['Re-clustering', 'No Re-clustering']
    
    for dtype in types:
        print(f"\n{'='*50}")
        print(f"[{category}] Experiment: Type={dtype}")
        print(f"{'='*50}")
        
        # 1. Prepare Data
        try:
            base_datasets = prepare_datasets(category, dtype)
        except Exception as e:
            print(f"[{category}] Skipping {dtype}: {e}")
            import traceback
            traceback.print_exc()
            continue
        
        for method in methods:
            if dtype == 'Continuous' and method == 'Re-clustering':
                print(f"  [{category}] Skipping {dtype} + {method} (Already done previously).")
                continue

            print(f"\n[{category}] --- Method: {method} ---")
            
            # Copy datasets
            curr_datasets = {k: v.copy() for k, v in base_datasets.items()}
            
            # 2. Apply "Clustering" Strategy
            try:
                if method == 'Re-clustering':
                    curr_datasets, K = run_optimized_clustering_pipeline(curr_datasets, category_name=f"{category}-{dtype}")
                else: 
                    curr_datasets, K = run_jtitle_clustering(curr_datasets)
                
                print(f"  [{category}] K = {K}")
                
                # 3. Format for SHN
                shn_data = prepare_shn_input(curr_datasets, category)
                
                # 4. Train SHN
                print(f"  [{category}] Starting Training...")
                
                model = train_shn(shn_data, K, gpu_id=gpu_id, epochs=50, warmup_epochs=5)
                
                # 5. Evaluate
                print(f"  [{category}] Evaluating...")
                y_test = shn_data['test']['y']
                preds, gate_probs = predict_shn(model, shn_data['test']['dense'], shn_data['test']['cluster'], gpu_id=gpu_id)
                
                metrics = evaluate_metrics(y_test, preds, y_prob=gate_probs)
                
                res = {
                    'Category': category,
                    'Dataset': dtype,
                    'Method': method,
                    'K': K,
                    'MAE': metrics['Global MAE'],
                    'Gate F1': metrics['Gate F1'],
                    'Gate Rec': metrics['Gate Rec'],
                    'Surv MAE': metrics['Surv MAE']
                }
                results_summary.append(res)
                print(f"  [{category}] Result: {res}")
            
            except Exception as e:
                print(f"  [{category}] Error during {method}: {e}")
                import traceback
                traceback.print_exc()

    # Save Results for this category
    df_res = pd.DataFrame(results_summary)
    out_csv = os.path.join(RESULTS_DIR, f"ablation_summary_{category}.csv")
    df_res.to_csv(out_csv, index=False)
    print(f"\n[{category}] All experiments done. Results saved to {out_csv}")
    print(df_res)


def run_ablation_experiment():
    # Detect GPU
    available_gpus = get_available_gpus()
    print(f"Main Process: Detected GPUs: {available_gpus}")
    
    categories = ['刑事', '民事', '行政']
    processes = []
    
    # Spawn a process for each category
    for i, cat in enumerate(categories):
        # Round-robin assignment of GPUs
        gpu_id = available_gpus[i % len(available_gpus)] if available_gpus else 0
        
        p = multiprocessing.Process(
            target=run_category_ablation, 
            args=(cat, gpu_id)
        )
        p.start()
        processes.append(p)
        print(f"Main Process: Launched {cat} on GPU {gpu_id}")
        # Small delay to prevent race conditions on file reads
        time.sleep(2) 
    
    # Wait for all to finish
    for p in processes:
        p.join()
        
    print("Main Process: All Parallel Tasks Completed.")


if __name__ == "__main__":
    run_ablation_experiment()
