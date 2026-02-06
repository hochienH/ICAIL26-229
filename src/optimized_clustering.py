import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.cluster import KMeans
from scipy.stats import entropy
from tqdm import tqdm
import joblib

try:
    import faiss
    HAS_FAISS = True
except ImportError:
    HAS_FAISS = False

class FaissKMeans:
    def __init__(self, n_clusters=8, n_init=10, max_iter=300, random_state=42, use_gpu=True):
        self.n_clusters = n_clusters
        self.n_init = n_init
        self.max_iter = max_iter
        self.random_state = random_state
        self.use_gpu = use_gpu
        self.cluster_centers_ = None
        self.inertia_ = None
        self.labels_ = None
        self.obj = None

    def fit(self, X):
        if not HAS_FAISS:
             raise ImportError("Faiss not installed")
        
        X = np.ascontiguousarray(X).astype('float32')
        d = X.shape[1]
        
        # Use simple faiss.Kmeans
        # Note: faiss.Kmeans uses 'seed' for random state if available in newer versions, 
        # but often it's global.
        kmeans = faiss.Kmeans(d, self.n_clusters, niter=self.max_iter, nredo=self.n_init, verbose=False, gpu=self.use_gpu, seed=self.random_state)
        kmeans.train(X)
        
        self.cluster_centers_ = kmeans.centroids
        self.inertia_ = kmeans.obj[-1] if hasattr(kmeans, 'obj') and len(kmeans.obj) > 0 else 0
        self.obj = kmeans
        
        # For labels, we need to search
        index = faiss.IndexFlatL2(d)
        if self.use_gpu:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            
        index.add(self.cluster_centers_)
        _, labels = index.search(X, 1)
        self.labels_ = labels.flatten()
        return self

    def fit_predict(self, X):
        self.fit(X)
        return self.labels_

    def predict(self, X):
        if self.cluster_centers_ is None:
            raise ValueError("Model not fitted yet")
            
        X = np.ascontiguousarray(X).astype('float32')
        d = X.shape[1]
        
        index = faiss.IndexFlatL2(d)
        if self.use_gpu and HAS_FAISS:
             res = faiss.StandardGpuResources()
             index = faiss.index_cpu_to_gpu(res, 0, index)

        index.add(self.cluster_centers_)
        _, labels = index.search(X, 1)
        return labels.flatten()

# ==========================================
# 1. 資料集讀取 (Updated for New Schema)
# ==========================================

class JudgmentDataset(Dataset):
    def __init__(self, parquet_paths, target_category=None):
        """
        讀取 Parquet 並解析 Dense Vector 與 JTITLE
        Args:
            parquet_paths: 單個檔案路徑或檔案路徑列表
            target_category: (Optional) 指定只處理 '刑事', '民事' 或 '行政'
        """
        if isinstance(parquet_paths, str):
            parquet_paths = [parquet_paths]
            
        print(f"Loading datasets from {len(parquet_paths)} files...")
        
        # 讀取並合併所有 parquet
        df_list = []
        for p in parquet_paths:
            if os.path.exists(p):
                df_temp = pd.read_parquet(p)
                df_list.append(df_temp)
            else:
                print(f"Warning: File not found {p}")
        
        if not df_list:
            raise ValueError("No valid parquet files loaded.")
            
        self.df = pd.concat(df_list, ignore_index=True)
        
        # 過濾特定類別 (如果需要)
        if target_category:
            print(f"Filtering for category: {target_category}")
            self.df = self.df[self.df['category'] == target_category].reset_index(drop=True)
            
        print(f"Total records loaded: {len(self.df)}")
        
        # [DEBUG] Removed sampling limit for production
        # if len(self.df) > 2000:
        #     print("⚠️ Sampling only 2000 records for quick testing...")
        #     self.df = self.df.sample(2000, random_state=42).reset_index(drop=True)
        
        print(f"Total records loaded: {len(self.df)}")
        
        # 1. 處理 Dense Vector
        # 檢查向量格式，確保是 numpy matrix (N, 1024)
        print("Processing dense vectors...")
        # 假設 parquet 讀出來是 numpy array 或 list
        first_vec = self.df['dense_vec'].iloc[0]
        if isinstance(first_vec, (list, np.ndarray)):
            # stack 會將 list of arrays 轉成 matrix
            self.vectors = np.stack(self.df['dense_vec'].values)
        else:
            raise ValueError("Format Error: 'dense_vec' column format is invalid.")

        # 2. 設定關鍵欄位
        # 使用 jtitle 作為案由標籤 (COA Label)
        self.df['coa_label'] = self.df['jtitle'].astype(str)
        
        # 使用 disputability 作為代表性採樣的依據 (Target Mean)
        # 確保它是數值型別
        self.df['disputability'] = pd.to_numeric(self.df['disputability'], errors='coerce').fillna(0)
        
        # 3. 建立 COA ID 映射
        self.coa_unique = self.df['coa_label'].unique()
        self.coa_to_id = {label: idx for idx, label in enumerate(self.coa_unique)}
        self.df['coa_id'] = self.df['coa_label'].map(self.coa_to_id)
        
        print(f"Dataset ready: {len(self.df)} cases, {len(self.coa_unique)} unique JTITLEs.")

    def get_coa_statistics(self):
        """
        計算每個案由的平均 Disputability (Methodology 4.2.2)
        這是 Representative Sampling 的基準：我們要選出最接近這個平均值的案件
        """
        # 過濾掉樣本數過少的案由，避免統計偏差 (這裡設為至少要有 5 筆)
        counts = self.df['coa_label'].value_counts()
        valid_coas = counts[counts >= 5].index
        
        stats = self.df[self.df['coa_label'].isin(valid_coas)].groupby('coa_label')['disputability'].mean().to_dict()
        return stats

# ==========================================
# 2. 核心演算法：Representative Sampling & Dual-Objective Clustering
# ==========================================

class SemanticClusteringManager:
    def __init__(self, dataset, m_support=100):
        self.dataset = dataset
        self.m = m_support
        self.prototypes = []        # 選出的代表性向量 (Sampled Vectors)
        self.prototype_coa_ids = [] # 對應的案由 ID
        
    def step1_representative_sampling(self):
        """
        Step 1: Representative Sampling
        選出最接近母體平均 Disputability 的 m 個案件作為 Support Set
        """
        print("\n=== Step 1: Performing Representative Sampling ===")
        pop_means = self.dataset.get_coa_statistics()
        grouped = self.dataset.df.groupby('coa_label')
        
        selected_vectors = []
        selected_coa_ids = []
        skipped_coas = 0
        
        # 遍歷每個案由 (JTITLE)
        for coa, group in tqdm(grouped, desc="Sampling COAs"):
            # 如果該案由不在統計名單內 (樣本太少)，直接跳過或全取
            if coa not in pop_means:
                # 策略：如果樣本極少 (< 5)，我們可以選擇忽略，或是全取
                # 這裡為了 Prototype 品質，選擇忽略極稀疏案由
                skipped_coas += 1
                continue

            target_mean = pop_means[coa]
            candidates = group.index.tolist()
            
            # 如果該案由總數就少於 m (例如只有 8 筆)，直接全部納入
            if len(group) <= self.m:
                chosen_indices = candidates
            else:
                # [演算法] 隨機抽樣 50 次，選 mean 最接近 target_mean 的那一組
                best_sample = None
                min_diff = float('inf')
                
                # Monte Carlo Approximation for "argmin"
                for _ in range(50):
                    sample_indices = np.random.choice(candidates, self.m, replace=False)
                    sample_vals = self.dataset.df.loc[sample_indices, 'disputability'].values
                    sample_mean = np.mean(sample_vals)
                    diff = abs(sample_mean - target_mean)
                    
                    if diff < min_diff:
                        min_diff = diff
                        best_sample = sample_indices
                
                chosen_indices = best_sample
            
            # 收集向量
            vecs = self.dataset.vectors[chosen_indices]
            selected_vectors.append(vecs)
            
            # 記錄這些向量屬於哪個 COA ID
            coa_id = self.dataset.coa_to_id[coa]
            selected_coa_ids.extend([coa_id] * len(chosen_indices))
            
        self.prototypes = np.vstack(selected_vectors)
        self.prototype_coa_ids = np.array(selected_coa_ids)
        print(f"Sampling Complete.")
        print(f"Total Prototype Vectors: {len(self.prototypes)}")
        print(f"Skipped COAs (too few samples): {skipped_coas}")
        
    def step2_dual_objective_grid_search(self, k_min=10, k_max=100, step=5, lambda_div=0.1):
        """
        Step 2: Grid Search for Optimal K
        Minimize Total Loss = Cohesion Loss + lambda * Diversity Penalty
        """
        print("\n=== Step 2: Dual-Objective Grid Search for K ===")
        print(f"Search Range: {k_min} to {k_max}, Lambda: {lambda_div}")
        
        results = []
        best_k = -1
        best_loss = float('inf')
        best_model = None
        
        k_range = range(k_min, k_max + 1, step)
        
        for k in k_range:
            # 1. 執行 K-Means
            kmeans = KMeans(n_clusters=k, n_init=10, random_state=42)
            labels = kmeans.fit_predict(self.prototypes)
            
            # 2. 計算 Cohesion Loss (希望同 JTITLE 在一起)
            cohesion_scores = []
            unique_coas = np.unique(self.prototype_coa_ids)
            
            for coa_id in unique_coas:
                mask = (self.prototype_coa_ids == coa_id)
                member_labels = labels[mask]
                
                if len(member_labels) > 0:
                    counts = np.bincount(member_labels)
                    n_max = counts.max()
                    n_total = len(member_labels) 
                    cohesion_scores.append(n_max / n_total)
            
            loss_cohesion = 1.0 - np.mean(cohesion_scores)
            
            # 3. 計算 Diversity Penalty (希望 Cluster 不要太雜)
            entropy_scores = []
            total_samples = len(self.prototypes)
            
            for cluster_id in range(k):
                mask = (labels == cluster_id)
                member_coas = self.prototype_coa_ids[mask]
                
                if len(member_coas) > 0:
                    value_counts = pd.Series(member_coas).value_counts(normalize=True)
                    ent = entropy(value_counts.values)
                    weight = len(member_coas) / total_samples
                    entropy_scores.append(weight * ent)
            
            loss_diversity = np.sum(entropy_scores)
            
            # 4. 總分
            total_loss = loss_cohesion + lambda_div * loss_diversity
            
            print(f"K={k:3d} | Total={total_loss:.4f} (Coh={loss_cohesion:.4f}, Div={loss_diversity:.4f})")
            
            results.append({
                'k': k, 'total': total_loss, 'coh': loss_cohesion, 'div': loss_diversity
            })
            
            if total_loss < best_loss:
                best_loss = total_loss
                best_k = k
                best_model = kmeans

        print(f"\n🏆 Best K found: {best_k} (Loss: {best_loss:.4f})")
        return best_model, best_k, pd.DataFrame(results)

# ==========================================
# 3. 執行流程
# ==========================================

def main():
    # 設定參數
    # Use paths relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data", "datasets_parquet")

    # 自動尋找資料集檔案
    parquet_files = []
    # 預設尋找刑事
    target_file = os.path.join(data_dir, "刑事_continuous.parquet")
    if os.path.exists(target_file):
        parquet_files.append(target_file)
    else:
        # 如果找不到，嘗試列出目錄下所有 continuous.parquet
        if os.path.exists(data_dir):
            for f in os.listdir(data_dir):
                if f.endswith("_continuous.parquet"):
                    parquet_files.append(os.path.join(data_dir, f))
    
    if not parquet_files:
        print(f"Warning: No parquet files found in {data_dir}")

    # 這裡演示只跑刑事 (若檔案列表有變，需相應調整)
    target_category = "刑事" 
    
    output_dir = "clustering_results"
    os.makedirs(output_dir, exist_ok=True)
    
    # 1. 載入資料 (自動讀取 jtitle 與 disputability)
    # 如果檔案不存在，請確保路徑正確
    try:
        dataset = JudgmentDataset(parquet_files, target_category=target_category)
    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # 2. 初始化分群管理器
    # m_support=100: 每個案由最多取 100 個樣本
    manager = SemanticClusteringManager(dataset, m_support=100)
    
    # 3. Step 1: Representative Sampling (使用 disputability 作為基準)
    manager.step1_representative_sampling()
    
    # 4. Step 2: Grid Search K
    # 根據取樣後的案由數量動態決定搜尋範圍
    n_sampled_coas = len(np.unique(manager.prototype_coa_ids))
    print(f"Number of JTITLEs used for clustering: {n_sampled_coas}")
    
    if n_sampled_coas < 5:
        print("Not enough COAs to perform clustering.")
        return

    # 搜尋範圍建議：從 COA 數量的 5% 到 60%
    k_min = max(2, int(n_sampled_coas * 0.05))
    k_max = min(n_sampled_coas, int(n_sampled_coas * 0.6))
    step = max(1, (k_max - k_min) // 10)
    
    best_kmeans, best_k, result_log = manager.step2_dual_objective_grid_search(
        k_min=k_min, k_max=k_max, step=step, lambda_div=0.1
    )
    
    # 5. 應用最佳模型：為所有資料分配 Cluster ID
    print("\nAssigning Cluster IDs to the entire dataset...")
    # 由於資料量可能很大 (10萬筆+)，分批預測以防 OOM (Optional but recommended)
    batch_size = 5000
    all_vectors = dataset.vectors
    all_cluster_ids = []
    
    for i in range(0, len(all_vectors), batch_size):
        batch = all_vectors[i : i + batch_size]
        ids = best_kmeans.predict(batch)
        all_cluster_ids.extend(ids)
        
    # 將結果存回 DataFrame
    dataset.df['cluster_id'] = all_cluster_ids
    
    # 6. 儲存結果
    # (A) 儲存處理好的 Dataframe (含 cluster_id, jtitle, disputability)
    # 建議加上 category 前綴以免混淆
    output_name = f"{target_category}_clustered.parquet" if target_category else "all_clustered.parquet"
    output_parquet = os.path.join(output_dir, output_name)
    dataset.df.to_parquet(output_parquet)
    print(f"✅ Saved clustered dataset to: {output_parquet}")
    
    # (B) 儲存 KMeans 模型 (包含 Centroids)
    model_path = os.path.join(output_dir, f"kmeans_model_{target_category}.joblib")
    joblib.dump(best_kmeans, model_path)
    print(f"✅ Saved KMeans model to: {model_path}")
    
    # (C) 儲存 Search Log
    log_path = os.path.join(output_dir, "grid_search_log.csv")
    result_log.to_csv(log_path, index=False)
    
    print("\nPre-processing Complete!")

if __name__ == "__main__":
    main()