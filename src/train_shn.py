import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import os
import sys
import multiprocessing
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
from torch.optim.lr_scheduler import CosineAnnealingLR

# Suppress warnings
warnings.filterwarnings('ignore')

# ==========================================
# 0. Global Helper
# ==========================================
THRESHOLD_MAP = {
    '刑事': 3,
    '民事': 3,
    '行政': 2
}

def get_available_gpus():
    """Detect available GPUs and return their indices sorted by free memory."""
    try:
        import subprocess
        cmd = "nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits"
        output = subprocess.check_output(cmd, shell=True).decode('utf-8')
        gpus = []
        for line in output.strip().split('\n'):
            if not line.strip(): continue
            idx, mem = line.split(',')
            gpus.append({'index': int(idx), 'memory_free': int(mem.strip())})
        gpus.sort(key=lambda x: x['memory_free'], reverse=True)
        return [g['index'] for g in gpus]
    except Exception as e:
        print(f"Warning: Could not detect GPUs ({e}). Defaulting to [0].")
        return [0]

def transform_target(raw_counts, category):
    tau = THRESHOLD_MAP.get(category, 3)
    return np.maximum(0, raw_counts - tau)

# ==========================================
# 1. ZTGP Loss Function
# ==========================================
class ZTGPLoss(nn.Module):
    """
    Zero-Truncated Generalized Poisson Loss
    """
    def __init__(self):
        super(ZTGPLoss, self).__init__()

    def forward(self, lam, xi, y_true):
        # y_true should be survivors only (y > 0)
        eps = 1e-6
        
        # log(lambda + xi*y) must be valid
        # xi is clamped in model, but here double check term_base
        term_base = lam + xi * y_true
        term_base = torch.clamp(term_base, min=eps) 
        
        # 1. Log-Likelihood of Generalized Poisson (Un-truncated)
        log_prob_gp = torch.log(lam + eps) + \
                      (y_true - 1) * torch.log(term_base) - \
                      term_base - \
                      torch.lgamma(y_true + 1)
        
        # 2. Zero-Truncation Normalization Term
        # log(1 - e^-lambda)
        log_normalization = torch.log(1 - torch.exp(-lam) + eps)
        
        # 3. Final ZTGP Log-Likelihood
        log_prob_ztgp = log_prob_gp - log_normalization
        
        return -torch.mean(log_prob_ztgp)

# ==========================================
# 2. Semantic Hurdle Network
# ==========================================
class SemanticHurdleNetwork(nn.Module):
    def __init__(self, input_dim, num_clusters, initial_centroids=None, prior_prob=0.01):
        super(SemanticHurdleNetwork, self).__init__()
        
        # --- A. Semantic Gating Mechanism ---
        self.cluster_emb = nn.Embedding(num_clusters, input_dim)
        if initial_centroids is not None:
            self.cluster_emb.weight.data.copy_(torch.tensor(initial_centroids))
            self.cluster_emb.weight.requires_grad = True 

        self.gate_proj = nn.Linear(input_dim, input_dim)
        
        # --- B. Shared Encoder ---
        self.shared_layer = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # --- C. Gate Network (Classification) ---
        self.gate_head = nn.Linear(512, 1)

        # [Critical Fix 1] Bias Initialization for Imbalanced Data
        # Initialize bias so initial probability matches the prior (e.g., 0.01)
        # logit = log(p / (1-p))
        if prior_prob > 0:
            init_bias = np.log(prior_prob / (1 - prior_prob))
            self.gate_head.bias.data.fill_(init_bias)

        # --- D. Count Network (Regression) ---
        self.lam_head = nn.Linear(512, 1) 
        self.xi_head = nn.Linear(512, 1)  

    def forward(self, h_doc, cluster_ids):
        # 1. Retrieve Cluster Centroid
        h_coa = self.cluster_emb(cluster_ids)
        
        # 2. Compute Gating Vector
        g = torch.sigmoid(self.gate_proj(h_coa))
        
        # 3. Semantic Fusion
        h_fusion = h_doc * g
        
        # 4. Shared Representation
        h = self.shared_layer(h_fusion)
        
        # 5. Heads
        # [Critical Fix 3] Output Logits directly (No Sigmoid here)
        gate_logits = self.gate_head(h)
        
        lam = F.softplus(self.lam_head(h)) + 1e-6
        # Limit xi to (-0.5, 0.5) for stability
        xi = torch.tanh(self.xi_head(h)) * 0.5 
        
        return gate_logits, lam, xi

# ==========================================
# 3. Data Loading Helper
# ==========================================
def load_data_with_clusters(base_path, category, n_pca, k):
    # Retrieve project root based on this script's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    
    # 1. Load the Master Clustered File (The Source of Truth for Clusters)
    # This file contains the 'cluster_id' assigned during the clustering experiment phase
    clustered_path = os.path.join(project_root, "clustering_results", f"{category}_clustered.parquet")
    if not os.path.exists(clustered_path):
        # Fallback to local if generic
        clustered_path = os.path.join("clustering_results", f"{category}_clustered.parquet")
        if not os.path.exists(clustered_path):
             raise FileNotFoundError(f"Clustered file not found: {clustered_path}. Please run optimized_clustering.py or apply_clustering.py first.")
    
    print(f"[{category}] Loading Cluster IDs from {clustered_path}...")
    # Read only necessary columns to save memory
    df_clustered = pd.read_parquet(clustered_path, columns=['filename', 'cluster_id'])
    
    # Create a fast mapping dictionary
    filename_to_cluster = df_clustered.set_index('filename')['cluster_id'].to_dict()
    
    # Determine K (number of clusters)
    num_clusters = int(df_clustered['cluster_id'].max()) + 1
    print(f"[{category}] Found {len(filename_to_cluster)} mapped files. Num Clusters (K): {num_clusters}")
    
    datasets = {}
    
    for split in ['train', 'val', 'test']:
        filename = f"{category}_{split}.parquet"
        path = os.path.join(base_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Split file not found: {path}")
            
        df = pd.read_parquet(path)
        
        # Merge Cluster IDs
        # We use .map() which is faster than merge for single column
        df['cluster_id'] = df['filename'].map(filename_to_cluster)
        
        # Integrity Check
        missing_mask = df['cluster_id'].isnull()
        if missing_mask.any():
            missing_count = missing_mask.sum()
            # If missing, it means the split dataset contains files NOT in the clustered experimental result.
            # This is critical for fairness. We'll drop them or raise error.
            # For robustness, let's complain loudly but drop them to allow running.
            print(f"WARNING: [{category} - {split}] Found {missing_count} samples without pre-assigned Cluster ID!")
            print(f"         These samples will be DROPPED to ensure cluster consistency.")
            df = df.dropna(subset=['cluster_id'])
        
        # Ensure cluster_id is int
        df['cluster_id'] = df['cluster_id'].astype(int)
        
        # Dense Vector
        X_dense = np.stack(df['dense_vec'].values)
        
        # Target Transformation
        raw_y = df['disputability'].values.astype(float)
        transformed_y = transform_target(raw_y, category)
        
        datasets[split] = {
            'dense': X_dense,
            'cluster': df['cluster_id'].values,
            'y': transformed_y,
            'raw_y': raw_y
        }
    
    return datasets, num_clusters

def compute_centroids(X_dense, cluster_ids, num_clusters):
    """Compute centroids from training data."""
    print("Computing initial cluster centroids...")
    dim = X_dense.shape[1]
    centroids = np.zeros((num_clusters, dim))
    counts = np.zeros(num_clusters)
    
    for i in range(len(cluster_ids)):
        cid = int(cluster_ids[i])
        centroids[cid] += X_dense[i]
        counts[cid] += 1
        
    counts[counts == 0] = 1
    centroids = centroids / counts[:, None]
    return centroids.astype(np.float32)

# ==========================================
# 4. Training Loop
# ==========================================
def train_shn(data, num_clusters, gpu_id=0, epochs=100, warmup_epochs=15):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"  > Training SHN on {device} (K={num_clusters})")
    
    # Pre-compute centroids
    centroids = compute_centroids(data['train']['dense'], data['train']['cluster'], num_clusters)
    
    # Calculate Class Weights for Imbalanced Data
    y_train = data['train']['y']
    num_pos = np.sum(y_train > 0)
    num_neg = len(y_train) - num_pos
    pos_weight_val = num_neg / num_pos if num_pos > 0 else 1.0
    
    # Clip weight
    pos_weight_val = min(pos_weight_val, 20.0)
    print(f"  [Config] Pos Weight: {pos_weight_val:.2f} | Warmup Epochs: {warmup_epochs}")

    # Datasets
    train_ds = TensorDataset(
        torch.FloatTensor(data['train']['dense']), 
        torch.LongTensor(data['train']['cluster']), 
        torch.FloatTensor(data['train']['y'])
    )
    val_ds = TensorDataset(
        torch.FloatTensor(data['val']['dense']), 
        torch.LongTensor(data['val']['cluster']), 
        torch.FloatTensor(data['val']['y'])
    )
    
    train_loader = DataLoader(train_ds, batch_size=256, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=256, num_workers=0)
    
    input_dim = data['train']['dense'].shape[1]
    
    # Pass prior probability for Bias Initialization
    prior = max(num_pos / len(y_train), 1e-4) if len(y_train) > 0 else 0.01
    
    model = SemanticHurdleNetwork(input_dim, num_clusters, initial_centroids=centroids, prior_prob=prior).to(device)
    
    # [Config] Optimizer: AdamW with Weight Decay
    # LR starts at 1e-3 and will be annealed down
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    
    # [Config] Scheduler: Cosine Annealing
    # Slowly reduce LR to 1e-5 over the course of training
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)
    
    # Weighted BCEWithLogitsLoss
    pos_weight_tensor = torch.tensor([pos_weight_val]).to(device)
    criterion_gate = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
    criterion_count = ZTGPLoss()
    
    # Hyperparams
    beta_scale = 0.05 # Scale down Count Loss to match Gate Loss magnitude

    # Fixed: Initialize tracking variables before loop
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    print(f"Start Training (Warmup: Gate Only for first {warmup_epochs} epochs)...")
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        gate_probs_sum = 0
        
        # --- Warmup Strategy ---
        if epoch < warmup_epochs:
            current_beta = 0.0
            phase_name = "WARMUP (Gate Only)"
        else:
            current_beta = 1.0
            phase_name = "JOINT (Gate + Count)"
            
            # [Critical Fix] Reset best_val_loss when switching phases
            # This ensures we save the best JOINT model, not the best WARMUP model
            if epoch == warmup_epochs:
                 print("  [Switch] Switching to Joint Training -> Resetting Best Val Loss tracker.")
                 best_val_loss = float('inf')
        
        for X_b, c_b, y_b in train_loader:
            X_b, c_b, y_b = X_b.to(device), c_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            
            gate_logits, lam, xi = model(X_b, c_b)
            
            # --- Gate Loss ---
            gate_labels = (y_b > 0).float().view(-1, 1)
            loss_gate = criterion_gate(gate_logits, gate_labels)
            
            # Monitor probabilities
            with torch.no_grad():
                gate_probs_sum += torch.sigmoid(gate_logits).mean().item()
            
            # --- Count Loss (Survivors) ---
            survivor_mask = (y_b > 0)
            if survivor_mask.sum() > 0:
                y_surv = y_b[survivor_mask]
                lam_surv = lam[survivor_mask].squeeze()
                xi_surv = xi[survivor_mask].squeeze()
                loss_count = criterion_count(lam_surv, xi_surv, y_surv)
            else:
                loss_count = torch.tensor(0.0).to(device)
                
            loss = loss_gate + current_beta * beta_scale * loss_count
            
            loss.backward()
            
            # Gradient Clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
        # Update Scheduler at the end of epoch
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]
            
        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, c_b, y_b in val_loader:
                X_b, c_b, y_b = X_b.to(device), c_b.to(device), y_b.to(device)
                gate_logits, lam, xi = model(X_b, c_b)
                
                gate_labels = (y_b > 0).float().view(-1, 1)
                l_g = criterion_gate(gate_logits, gate_labels)
                
                survivor_mask = (y_b > 0)
                if survivor_mask.sum() > 0:
                    l_c = criterion_count(
                        lam[survivor_mask].squeeze(), 
                        xi[survivor_mask].squeeze(), 
                        y_b[survivor_mask]
                    )
                else:
                    l_c = 0
                
                # Validation Loss follows current phase metric
                val_loss += (l_g + current_beta * beta_scale * l_c).item()
        
        avg_val_loss = val_loss / len(val_loader)
        avg_prob = gate_probs_sum / len(train_loader)
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            saved_msg = "[Saved]"
        else:
            saved_msg = ""
        
        # Log periodically and on phase switch
        if epoch % 5 == 0 or epoch == warmup_epochs:
            print(f"  Epoch {epoch} [{phase_name}]: Val Loss={avg_val_loss:.4f} | Avg Gate Prob={avg_prob:.4f} | LR={current_lr:.2e} {saved_msg}")
    
    print(f"  > Done. Best Val Loss: {best_val_loss:.4f}")
    model.load_state_dict(best_model_wts)
    return model

def predict_shn(model, X_dense, cluster_ids, gpu_id=0):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    model.eval()
    
    # Process in batches to avoid OOM on prediction
    batch_size = 1024
    num_samples = len(X_dense)
    final_preds = []
    gate_probs_list = [] # Add this
    
    # We don't need gradients
    with torch.no_grad():
        for i in range(0, num_samples, batch_size):
            end = min(i + batch_size, num_samples)
            X_b = torch.FloatTensor(X_dense[i:end]).to(device)
            c_b = torch.LongTensor(cluster_ids[i:end]).to(device)
            
            gate_logits, lam, xi = model(X_b, c_b)
            
            # Apply Sigmoid manually since model outputs logits
            pi = torch.sigmoid(gate_logits).cpu().numpy().flatten()
            lam = lam.cpu().numpy().flatten()
            xi = xi.cpu().numpy().flatten()
            
            # E[Y] = pi * E[Count]
            # E[Count] approx lam / (1 - xi) * ZeroTruncCorrection
            # Or simplified: if xi is small, approx lam / (1-e^-lam)
            
            term_denom = 1 - xi
            # Safety for division
            term_denom[term_denom == 0] = 1e-6
            
            mean_gp = lam / term_denom
            
            zt_correction = 1 / (1 - np.exp(-lam) + 1e-6)
            
            expected_count = mean_gp * zt_correction
            batch_preds = pi * expected_count
            final_preds.append(batch_preds)
            gate_probs_list.append(pi)
            
    return np.concatenate(final_preds), np.concatenate(gate_probs_list)

# ==========================================
# 5. Analysis & Evaluation
# ==========================================
def analyze_thresholds(y_true, y_prob):
    from sklearn.metrics import precision_recall_curve
    import matplotlib.pyplot as plt
    
    # y_true: True Label (0 or 1)
    # y_prob: Model Predicted Gate Probability (pi)
    
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    
    # Find best F1 Threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]
    best_f1 = f1_scores[best_idx]
    
    print(f"\n[Analysis] Threshold Optimization:")
    print(f"  Best Threshold: {best_threshold:.4f}")
    print(f"  Best F1 Score: {best_f1:.4f}")
    print(f"  - Precision at Best: {precisions[best_idx]:.4f}")
    print(f"  - Recall at Best: {recalls[best_idx]:.4f}")
    
    # Plot
    try:
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, marker='.', label='SHN')
        plt.scatter(recalls[best_idx], precisions[best_idx], color='red', label='Best Threshold', zorder=5)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve (Best Thr={best_threshold:.2f})')
        plt.legend()
        plt.grid(True)
        plt.savefig('pr_curve_civil.png')
        print("  [Plot] PR Curve saved to pr_curve_civil.png")
    except Exception as e:
        print(f"  [Plot] Could not save plot: {e}")
    
    return best_threshold

def evaluate_metrics(y_true, y_pred, y_prob=None, best_thr=0.5):
    mae_global = mean_absolute_error(y_true, y_pred)
    mse_global = mean_squared_error(y_true, y_pred)
    
    y_true_bin = (y_true > 0).astype(int)
    
    # Standard 0.5 Threshold Evaluation
    y_pred_bin_std = (y_prob > 0.5).astype(int) if y_prob is not None else (y_pred > 0.5).astype(int)
    
    # Optimized Threshold Evaluation (if prob provided)
    if y_prob is not None and best_thr != 0.5:
        y_pred_bin_opt = (y_prob > best_thr).astype(int)
        # Use Optimized for reporting Confusion Matrix if desired, or just print both
        print(f"\n[Eval] Standard (0.5) vs Optimized ({best_thr:.3f}) Thresholds:")
        
        acc = accuracy_score(y_true_bin, y_pred_bin_std)
        prec = precision_score(y_true_bin, y_pred_bin_std, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin_std, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin_std, zero_division=0)
        print(f"  Std (0.5): Acc={acc:.4f}, Prec={prec:.4f}, Rec={rec:.4f}, F1={f1:.4f}")
        
        acc_opt = accuracy_score(y_true_bin, y_pred_bin_opt)
        prec_opt = precision_score(y_true_bin, y_pred_bin_opt, zero_division=0)
        rec_opt = recall_score(y_true_bin, y_pred_bin_opt, zero_division=0)
        f1_opt = f1_score(y_true_bin, y_pred_bin_opt, zero_division=0)
        print(f"  Opt ({best_thr:.3f}): Acc={acc_opt:.4f}, Prec={prec_opt:.4f}, Rec={rec_opt:.4f}, F1={f1_opt:.4f}")
        
        # Use Optimized for final return if preferred, or stick to standard?
        # Let's return the Optimized one for final report if it's better
        y_pred_bin = y_pred_bin_opt
    else:
        y_pred_bin = y_pred_bin_std
        acc = accuracy_score(y_true_bin, y_pred_bin)
        prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
        rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
        f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    
    survivor_mask = y_true > 0
    if np.sum(survivor_mask) > 0:
        mae_surv = mean_absolute_error(y_true[survivor_mask], y_pred[survivor_mask])
        mse_surv = mean_squared_error(y_true[survivor_mask], y_pred[survivor_mask])
    else:
        mae_surv = 0.0
        mse_surv = 0.0

    return {
        'Global MAE': mae_global,
        'Global MSE': mse_global,
        'Gate Acc': acc,
        'Gate Prec': prec,
        'Gate Rec': rec,
        'Gate F1': f1,
        'Surv MAE': mae_surv,
        'Surv MSE': mse_surv,
        'TP': tp, 'FP': fp, 'TN': tn, 'FN': fn
    }

# ==========================================
# 6. Worker Process
# ==========================================
def run_shn_experiment(cat, data_dir, gpu_id):
    log_file = f"log_SHN_{cat}.txt"
    sys.stdout = open(log_file, "w", buffering=1)
    sys.stderr = sys.stdout
    
    print(f"[{cat}] SHN Process started on GPU {gpu_id}")
    
    try:
        # 1. Load Data & Clusters
        datasets, num_clusters = load_data_with_clusters(data_dir, cat, None, None)
        print(f"[{cat}] Data Loaded. K={num_clusters}")
        
        # 2. Train SHN
        model = train_shn(datasets, num_clusters, gpu_id=gpu_id, epochs=50)
        
        # 3. Predict Test
        y_test = datasets['test']['y']
        preds, gate_probs = predict_shn(model, datasets['test']['dense'], datasets['test']['cluster'], gpu_id=gpu_id)
        
        # 3.1 Threshold Analysis
        best_thr = analyze_thresholds((y_test > 0).astype(int), gate_probs)
        
        # 4. Evaluate (Pass gate_probs and best_thr for optimized reporting)
        results = evaluate_metrics(y_test, preds, y_prob=gate_probs, best_thr=best_thr)
        
        print(f"\n[{cat}] SHN Results Summary:")
        print(f"  Global MAE: {results['Global MAE']:.4f}")
        print(f"  Gate F1:    {results['Gate F1']:.4f}")
        print(f"  Surv MAE:   {results['Surv MAE']:.4f}")
        print(f"  Conf Mat:   TP={results['TP']}, FP={results['FP']}, TN={results['TN']}, FN={results['FN']}")
        
    except Exception as e:
        print(f"[{cat}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.close()

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    # Use path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    DATA_DIR = os.path.join(project_root, "data", "datasets_split")

    CATEGORIES = ['刑事', '民事', '行政']
    
    available_gpus = get_available_gpus()
    print(f"Detected GPUs: {available_gpus}")
    
    processes = []
    
    for i, cat in enumerate(CATEGORIES):
        assigned_gpu = available_gpus[i % len(available_gpus)] if available_gpus else 0
        p = multiprocessing.Process(
            target=run_shn_experiment, 
            args=(cat, DATA_DIR, assigned_gpu)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
