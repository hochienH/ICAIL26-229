import os
import sys
import numpy as np
import pandas as pd
import joblib
import multiprocessing
import subprocess # Added missing import

# Add current directory to path so we can import from optimized_clustering
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from optimized_clustering import JudgmentDataset, SemanticClusteringManager, FaissKMeans, HAS_FAISS


def apply_clustering_known_k(file_path, category, k, gpu_id):
    """
    直接使用已知的 K 值進行分群 (Skip Grid Search)
    """
    # 設定環境變數指定使用的 GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    
    # Redirect stdout to avoid interleaving
    log_file = f"log_{category}_apply.txt"
    sys.stdout = open(log_file, "w", buffering=1)
    sys.stderr = sys.stdout
    
    print(f"[{category}] Process Apply Started. GPU: {gpu_id}, K={k}")
    
    output_dir = "clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. Load Data
    try:
        dataset = JudgmentDataset(file_path, target_category=category)
    except Exception as e:
        print(f"[{category}] ❌ Error loading data: {e}")
        return

    # 2. Manager & Sampling (Sampling is still needed to train the model centers!)
    print(f"[{category}] Sampling prototypes for training...")
    manager = SemanticClusteringManager(dataset, m_support=100)
    manager.step1_representative_sampling()
    
    if len(manager.prototypes) < k:
        print(f"[{category}] Error: Not enough prototypes ({len(manager.prototypes)}) for K={k}")
        return

    # 3. Train Model directly
    print(f"[{category}] Training KMeans model with K={k}...")
    if HAS_FAISS:
        kmeans = FaissKMeans(n_clusters=k, n_init=10, random_state=42, use_gpu=True)
    else:
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
        
    kmeans.fit_predict(manager.prototypes)
    
    # 4. Predict Full Dataset
    print(f"[{category}] Assigning Cluster IDs to all {len(dataset.df)} records...")
    batch_size = 5000
    all_vectors = dataset.vectors
    all_cluster_ids = []
    
    for i in range(0, len(all_vectors), batch_size):
        batch = all_vectors[i : i + batch_size]
        ids = kmeans.predict(batch)
        all_cluster_ids.extend(ids)
        
    dataset.df['cluster_id'] = all_cluster_ids
    
    # 5. Save
    out_pq = os.path.join(output_dir, f"{category}_clustered.parquet")
    out_model = os.path.join(output_dir, f"kmeans_model_{category}.joblib")
    
    # User requested to exclude 'coa_label' and 'coa_id' from the final output
    cols_to_drop = ['coa_label', 'coa_id']
    df_to_save = dataset.df.drop(columns=[c for c in cols_to_drop if c in dataset.df.columns])
    
    df_to_save.to_parquet(out_pq)
    
    from sklearn.cluster import KMeans as SklearnKMeans

    # Handle Faiss Pickle Issue: Convert to Sklearn KMeans before saving
    if isinstance(kmeans, FaissKMeans):
        print(f"[{category}] Converting FaissKMeans to sklearn.cluster.KMeans for pickling...")
        sklearn_kmeans = SklearnKMeans(n_clusters=k, n_init=10, random_state=42)
        
        # Manually populate attributes needed for prediction/compatibility
        sklearn_kmeans.cluster_centers_ = kmeans.cluster_centers_
        sklearn_kmeans.labels_ = kmeans.labels_
        sklearn_kmeans._n_threads = 4  # Safe default to avoid thread issues on load
        
        # Faiss might store objective value (inertia)
        try:
             # Faiss Kmeans typically stores iteration stats, but just in case
             sklearn_kmeans.inertia_ = kmeans.obj.obj[-1] if kmeans.obj and len(kmeans.obj.obj) > 0 else 0
        except:
             sklearn_kmeans.inertia_ = 0
             
        # Manually set n_iter_ (dummy value if unknown)
        sklearn_kmeans.n_iter_ = 300 
        
        kmeans = sklearn_kmeans

    joblib.dump(kmeans, out_model)
    print(f"[{category}] ✅ Done. Saved to {out_pq}")
    
    
def main():
    # Use path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data", "datasets_parquet")
    
    # 定義要直接套用的任務與 K 值
    tasks = [
        {"file": os.path.join(data_dir, "刑事_continuous.parquet"), "category": "刑事", "k": 151},
        {"file": os.path.join(data_dir, "民事_continuous.parquet"), "category": "民事", "k": 114},
        {"file": os.path.join(data_dir, "行政_continuous.parquet"), "category": "行政", "k": 284},
    ]
    
    # 如果您只要跑刑事 K=64，請確認
    # tasks = [
    #      {"file": os.path.join(data_dir, "刑事_continuous.parquet"), "category": "刑事", "k": 64}
    # ]
    
    # GPU Selection (Better Logic: Find GPU with MAX FREE memory)
    best_gpu = 0 
    try:
        # Sort by memory.free DESCENDING (biggest free memory first)
        cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode('utf-8')
        gpus = []
        for line in output.strip().split('\n'):
            idx, mem = line.split(',')
            gpus.append({'index': int(idx), 'memory_free': int(mem.strip())})
        
        gpus.sort(key=lambda x: x['memory_free'], reverse=True)
        
        if gpus:
            best_gpu = gpus[0]['index']
            free_mem = gpus[0]['memory_free']
            print(f"   -> Detected Best GPU: {best_gpu} (Free Memory: {free_mem} MiB)")
    except Exception as e:
        print(f"   -> GPU Detection Failed: {e}. Defaulting to GPU 0.")
        pass
        
    print(f"🚀 Launching Apply Tasks on GPU {best_gpu} (Sequential Execution)...")
    
    for t in tasks:
        # Run sequentially to avoid OOM
        apply_clustering_known_k(t['file'], t['category'], t['k'], best_gpu)
        
    print("All tasks finished.")

if __name__ == "__main__":
    main()
