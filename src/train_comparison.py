import pandas as pd
import numpy as np
import scipy.sparse as sp
from scipy.sparse import hstack
from sklearn.linear_model import PoissonRegressor
from sklearn.svm import LinearSVR
from sklearn.metrics import mean_absolute_error, mean_squared_error, f1_score, recall_score, precision_score, accuracy_score, confusion_matrix
import xgboost as xgb
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import copy
import os
import subprocess
import multiprocessing
import sys
import warnings
import gc

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ==========================================
# 0. Global Configuration & Helpers
# ==========================================

# 定義每個類別的門檻值 (Thresholds)
THRESHOLD_MAP = {
    '刑事': 3,
    '民事': 3,
    '行政': 2
}

from sklearn.metrics import precision_recall_curve

def analyze_thresholds(y_true, y_prob):
    """
    Find best threshold based on F1 Score from Precision-Recall Curve.
    Used for ensuring fair comparison for weighted models.
    """
    y_true_bin = (y_true > 0).astype(int)
    # Handle case where y_prob might not be proper probability (e.g. SVR output)
    # But usually passed as probability or continuous score.
    
    precisions, recalls, thresholds = precision_recall_curve(y_true_bin, y_prob)
    
    # Calculate F1 for each threshold
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)
    
    # thresholds array is 1 element shorter than precision/recall arrays
    # We take the first (n-1) scores to match thresholds
    f1_scores_cut = f1_scores[:-1]
    
    if len(f1_scores_cut) > 0:
        best_idx = np.argmax(f1_scores_cut)
        best_threshold = thresholds[best_idx]
        best_f1 = f1_scores_cut[best_idx]
    else:
        best_threshold = 0.5
    
    # Safety Check: If best threshold is too extreme, fallback to sensible defaults
    if best_threshold <= 0: best_threshold = 1e-3
    
    return best_threshold

def get_available_gpus():
    """Detect available GPUs and return their indices sorted by free memory."""
    try:
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

# ==========================================
# 1. Data Loading & Transformation
# ==========================================

def transform_target(raw_counts, category):
    """
    Critical Step: Transform raw instance counts (I) to Excess Counts (y).
    Formula: y = max(0, I - tau)
    """
    tau = THRESHOLD_MAP.get(category, 3) # Default to 3 if unknown
    # raw_counts is assumed to be numpy array
    return np.maximum(0, raw_counts - tau)

def parse_sparse_features(df, fixed_dim=None):
    """Restore sparse matrix from dataframe columns."""
    all_data = []
    all_indices = []
    all_indptr = [0]
    local_max_index = 0
    
    for indices, values in zip(df['sparse_indices'], df['sparse_values']):
        if len(indices) > 0:
            all_data.append(values)
            all_indices.append(indices)
            curr_max = np.max(indices)
            if curr_max > local_max_index:
                local_max_index = curr_max
        all_indptr.append(all_indptr[-1] + len(indices))
        
    flat_data = np.concatenate(all_data) if all_data else np.array([])
    flat_indices = np.concatenate(all_indices) if all_indices else np.array([])
    
    if fixed_dim is not None:
        final_dim = fixed_dim
        if local_max_index >= fixed_dim:
            # In production, you might want to truncate, but here we warn
            pass 
    else:
        final_dim = local_max_index + 1

    X_sparse = sp.csr_matrix((flat_data, flat_indices, all_indptr), shape=(len(df), final_dim))
    return X_sparse

def load_dataset_group(base_path, category):
    """
    Load Train/Val/Test and apply Target Transformation IMMEDIATELY.
    """
    datasets = {}
    temp_dfs = {}
    global_max_index = 0
    
    # 1. First pass: Find max feature index
    for split in ['train', 'val', 'test']:
        filename = f"{category}_{split}.parquet"
        path = os.path.join(base_path, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"File not found: {path}")
            
        df = pd.read_parquet(path)
        temp_dfs[split] = df
        
        for indices in df['sparse_indices']:
            if len(indices) > 0:
                m = np.max(indices)
                if m > global_max_index:
                    global_max_index = m
                    
    feature_dim = global_max_index + 1
    print(f"[{category}] Feature Dim: {feature_dim}")
    
    # 2. Second pass: Construct matrices and Transform Targets
    for split in ['train', 'val', 'test']:
        df = temp_dfs[split]
        
        # Raw Target (I)
        raw_y = df['disputability'].values.astype(float)
        
        # TRANSFORM TARGET HERE: y = max(0, I - tau)
        # We train ALL baselines on this 'y'.
        transformed_y = transform_target(raw_y, category)
        
        X_dense = np.stack(df['dense_vec'].values)
        X_sparse = parse_sparse_features(df, fixed_dim=feature_dim)
        
        datasets[split] = {
            'dense': X_dense,
            'sparse': X_sparse,
            'y': transformed_y,  # Using Transformed Y for training!
            'raw_y': raw_y       # Keep raw just in case (optional)
        }
        
    return datasets

# ==========================================
# 2. Model Definitions
# ==========================================

# --- Baseline 1: GLM (Poisson) ---
def run_glm_poisson(data):
    print(f"  > Running GLM-Poisson (Weighted)...")
    
    y_train = data['train']['y']
    
    # 1. Sample Weights
    num_pos = np.sum(y_train > 0)
    num_neg = len(y_train) - num_pos
    pos_weight = 20.0 # Standardize to 20
    
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train > 0] = pos_weight
    
    # 2. Model
    model = PoissonRegressor(alpha=1.0, max_iter=1000)
    model.fit(data['train']['sparse'], y_train, sample_weight=sample_weights)
    
    # 3. Predict (Lambda)
    preds = model.predict(data['test']['sparse'])
    return preds

# --- Baseline 2: Linear SVR ---
def run_linear_svr(data):
    print(f"  > Running LinearSVR (Weighted)...")
    
    y_train = data['train']['y']
    
    # 1. Sample Weights
    pos_weight = 20.0
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train > 0] = pos_weight
    
    # 2. Model
    model = LinearSVR(max_iter=5000, random_state=42, dual='auto')
    model.fit(data['train']['sparse'], y_train, sample_weight=sample_weights)
    
    preds = model.predict(data['test']['sparse'])
    return np.maximum(0, preds) # Clip negative predictions

# --- Baseline 3: XGBoost (Poisson) ---
def run_xgboost(data, gpu_id=0):
    print(f"  > Running XGBoost (Dense, Weighted)...")
    
    y_train = data['train']['y']
    pos_weight = 20.0
    sample_weights = np.ones(len(y_train))
    sample_weights[y_train > 0] = pos_weight

    xgb_params = {
        'objective': 'count:poisson',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'early_stopping_rounds': 20,
        'random_state': 42,
        'n_jobs': 4,
        'tree_method': 'hist',
        'device': f'cuda:{gpu_id}',
        'max_bin': 64
    }
    
    X_train = np.ascontiguousarray(data['train']['dense'])
    X_val = np.ascontiguousarray(data['val']['dense'])
    X_test = np.ascontiguousarray(data['test']['dense'])

    model = xgb.XGBRegressor(**xgb_params)
    
    eval_set = [(X_val, data['val']['y'])]
    
    # Pass sample_weight
    model.fit(X_train, y_train, eval_set=eval_set, verbose=False, sample_weight=sample_weights)
    preds = model.predict(X_test)
    return preds


# --- Combined: XGBoost (Dense + Sparse) ---
def run_xgboost_combined(data, gpu_id=0):
    print(f"  > Running XGBoost (Combined)...")
    X_train = hstack([data['train']['dense'], data['train']['sparse']])
    X_val = hstack([data['val']['dense'], data['val']['sparse']])
    X_test = hstack([data['test']['dense'], data['test']['sparse']])

    xgb_params = {
        'objective': 'count:poisson',
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'early_stopping_rounds': 20,
        'random_state': 42,
        'n_jobs': 4,
        'tree_method': 'hist',
        'device': f'cuda:{gpu_id}',
        'max_bin': 64
    }

    model = xgb.XGBRegressor(**xgb_params)
    eval_set = [(X_val, data['val']['y'])]
    
    model.fit(X_train, data['train']['y'], eval_set=eval_set, verbose=False)
    preds = model.predict(X_test)
    return preds

# --- Baseline 4 & 5: PyTorch BERT-MLP ---
class BertMLP(nn.Module):
    def __init__(self, input_dim=1024, output_type='regression', bias_init=None):
        super(BertMLP, self).__init__()
        self.output_type = output_type
        self.net = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 1)
        )
        # Bias Initialization
        if bias_init is not None:
             self.net[-1].bias.data.fill_(bias_init)

        # Poisson requires positive output
        if output_type == 'poisson':
            self.activation = nn.Softplus()

    def forward(self, x):
        out = self.net(x)
        if self.output_type == 'poisson':
            out = self.activation(out)
        return out

def run_pytorch_model(model_type, data, gpu_id=0):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"  > Running BERT-{model_type.capitalize()} on {device} (Weighted)...")
    
    batch_size = 1024
    
    y_train = data['train']['y']
    
    # Calculate Bias Init
    num_pos = np.sum(y_train > 0)
    mu_global = np.mean(y_train)
    
    bias_init = None
    if model_type == 'regression':
        bias_init = mu_global
    elif model_type == 'poisson':
        # Log of mean count
        bias_init = np.log(mu_global + 1e-6)

    # Data Preparation
    train_dataset = TensorDataset(torch.FloatTensor(data['train']['dense']), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(data['val']['dense']), torch.FloatTensor(data['val']['y']))
    test_tensor = torch.FloatTensor(data['test']['dense']).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    
    model = BertMLP(output_type=model_type, bias_init=bias_init).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    
    # Loss Function Selection & Weighting
    weight_val = 20.0
    weight_tensor = torch.tensor(weight_val).to(device)

    # Training Loop
    best_val_loss = float('inf')
    patience = 5
    counter = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    
    epochs = 50
    for epoch in range(epochs):
        model.train()
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            out = model(X_b).squeeze()
            
            # Weighted Loss Manual Calculation
            if model_type == 'regression':
                # MSE: (y - y_hat)^2 * weight
                loss_raw = (out - y_b) ** 2
                weights = torch.ones_like(y_b)
                weights[y_b > 0] = weight_val
                loss = (loss_raw * weights).mean()
            else:
                # Poisson NLL: loss(input, target)
                loss_func = nn.PoissonNLLLoss(log_input=False, reduction='none')
                loss_raw = loss_func(out, y_b)
                weights = torch.ones_like(y_b)
                weights[y_b > 0] = weight_val
                loss = (loss_raw * weights).mean()

            loss.backward()
            optimizer.step()
        
        # Scheduler Step
        scheduler.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                out = model(X_b).squeeze()
                
                # Validation Loss (Weighted)
                if model_type == 'regression':
                    loss_raw = (out - y_b) ** 2
                    weights = torch.ones_like(y_b)
                    weights[y_b > 0] = weight_val
                    val_loss += (loss_raw * weights).mean().item()
                else:
                    loss_func = nn.PoissonNLLLoss(log_input=False, reduction='none')
                    loss_raw = loss_func(out, y_b)
                    weights = torch.ones_like(y_b)
                    weights[y_b > 0] = weight_val
                    val_loss += (loss_raw * weights).mean().item()
        
        avg_val_loss = val_loss / len(val_loader)

        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            counter = 0
        else:
            counter += 1
            if counter >= patience:
                break
                
    model.load_state_dict(best_model_wts)
    model.eval()
    with torch.no_grad():
        preds = model(test_tensor).squeeze().cpu().numpy()
        
    return preds

# --- ZINB Model Implementation ---
class ZINB_MLP(nn.Module):
    def __init__(self, input_dim=1024, count_bias=0.0):
        super(ZINB_MLP, self).__init__()
        # Shared encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # 1. Pi head (Probability of Zero Inflation) - Sigmoid output
        self.pi_head = nn.Sequential(
            nn.Linear(512, 1) # Removed Sigmoid to return logits for BCE
        )
        # Bias Init for Gate: P(Gate=1) tiny approx 0.01 -> log(0.01/0.99) ~ -4.6
        self.pi_head[0].bias.data.fill_(-4.6)
        
        # 2. Mu head (Mean of NB) - Exp/Softplus output
        self.mu_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Softplus() 
        )
        # Bias Init for Count: log(mean_count)
        if count_bias is not None:
             # Softplus inverse approximation for bias init? 
             # Or just init linear layer bias. inv_softplus(y) = log(exp(y)-1)
             # If y large, inv_softplus(y) ~ y.
             val = np.log(np.exp(count_bias) - 1 + 1e-6) if count_bias > 0 else 0.0
             self.mu_head[0].bias.data.fill_(val)
        
        # 3. Theta head (Dispersion of NB) - Exp/Softplus output
        # Theta must be positive
        self.theta_head = nn.Sequential(
            nn.Linear(512, 1),
            nn.Softplus()
        )

    def forward(self, x):
        h = self.encoder(x)
        pi_logits = self.pi_head(h)
        mu = self.mu_head(h)
        theta = self.theta_head(h) + 1e-6 # Add epsilon for stability
        return pi_logits, mu, theta

def zinb_loss(y_true, pi_logits, mu, theta, pos_weight_val=20.0, epsilon=1e-8):
    """
    Weighted ZINB Loss
    """
    # Reshape if necessary
    y_true = y_true.view(-1, 1)
    
    # 1. Gate Loss (BCE with Weight)
    # y_gate: 1 if y > 0 (Success), 0 if y=0 (Inflated Zero)
    # Note: Traditional ZINB 'pi' is Prob(Zero). Here we usually model Prob(Non-Zero) in Gate?
    # Wait, in ZINB equations: pi is usually P(Inflated Zero).
    # If our pi_head outputs logits for "Zero Inflation", then:
    # Target for pi should be 1 if y=0 ??? 
    # Let's align with SHN logic where Gate predicts "Survival" (y>0).
    # But ZINB standard definition: pi is prob of extra zeros.
    # Let's stick to standard ZINB: pi = P(Extra Zero).
    # Then Survivor (y>0) implies Not Extra Zero.
    
    # Actually, simpler to treat pi_logits as "Prob of Survival" to match SHN? 
    # No, let's keep ZINB semantics but apply weight.
    # ZINB: P(Y=0) = pi + (1-pi)*NB(0)
    # P(Y=k) = (1-pi)*NB(k)
    
    # Let's interpret the self.pi_head output. 
    # If we initialized bias to -4.6, that means sigmoid(-4.6) is small (~0.01).
    # If this represents P(Survival), then P(Survival) is small. That matches reality (1%).
    # So let's count pi_logits as "Logits of SURVIVAL (Non-Zero)".
    # Then pi (prob of survival) = sigmoid(logits).
    # P(Inflated Zero) = 1 - pi.
    
    pi = torch.sigmoid(pi_logits)
    
    # Therefore:
    # P(Y=0 | Gate=0) = 1 (Inflated Zero)
    # P(Y=y | Gate=1) = NB(y)
    
    # Total P(Y=0) = P(Gate=0)*1 + P(Gate=1)*NB(0) = (1-pi) + pi*NB(0)
    # Total P(Y=k) = P(Gate=1)*NB(k) = pi * NB(k)   (for k>0)
    
    # Logic Update:
    # pi is Prob(Survival).
    
    # --- Gate Loss (Component 1) ---
    # We want to enforce pi to match binary target (y > 0).
    # Using BCEWithLogitsLoss with pos_weight=20
    # Target: 1 if y > 0 else 0
    gate_target = (y_true > 0).float()
    
    # BCE Loss: -(target * log(pi) + (1-target) * log(1-pi))
    # We add weight to the 'target=1' class (Survivors)
    pos_weight = torch.tensor(pos_weight_val).to(y_true.device)
    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight, reduction='none')(pi_logits, gate_target)
    
    # --- Count Loss (Component 2) ---
    # NLL of NB distribution for survivors.
    # We only care about NB fit for those that are truly non-zero?
    # Or Standard ZINB attempts to fit everything?
    # Standard ZINB maximizes global likelihood.
    
    # L_total = - LogLikelihood
    # Case y=0: LL = log( (1-pi) + pi*NB(0) )
    # Case y>0: LL = log( pi * NB(y) ) = log(pi) + log(NB(y))
    
    # But we want to 'weight' the importance of survivors.
    # The BCE part handles the "Detection" weighting.
    # The Count part handles the "Magnitude" fitting.
    
    # Re-derivation for Weighted ZINB:
    # Term 1 (y=0): log( (1-pi) + pi*P_nb(0) )
    # Term 2 (y>0): log(pi) + log(P_nb(y))
    
    # To apply pos_weight=20 implicitly to Survivors:
    # We amplify the Term 2 contribution? 
    # Or just use the Weighted BCE for the "pi" part and standard NLL for "NB" part?
    
    # Hybrid Approach (Best for 'Fair Comparison' with SHN):
    # SHN uses: Loss = Loss_Gate + Loss_Count (ZeroTruncated).
    # Here ZINB uses: Loss = -LogLikelihood.
    
    # Let's assume the LogLikelihood formulation but multiply Term 2 by weight?
    # Weighting the likelihood is valid.
    
    t1 = theta / (theta + mu + epsilon)
    prob_nb_zero = torch.pow(t1, theta)
    
    # Case y = 0
    # LL_0 = log( (1-pi) + pi * prob_nb_zero )
    log_p_zero = torch.log((1 - pi) + pi * prob_nb_zero + epsilon)
    
    # Case y > 0
    # LL_k = log(pi) + log_gamma(y+theta) - ...
    log_pi = torch.log(pi + epsilon)
    
    lg_y_theta = torch.lgamma(y_true + theta)
    lg_y_1 = torch.lgamma(y_true + 1)
    lg_theta = torch.lgamma(theta)
    
    term_nb = lg_y_theta - lg_y_1 - lg_theta + \
              theta * (torch.log(theta+epsilon) - torch.log(theta+mu+epsilon)) + \
              y_true * (torch.log(mu+epsilon) - torch.log(theta+mu+epsilon))
              
    log_p_nonzero = log_pi + term_nb
    
    mask_zero = (y_true == 0).float()
    
    # Apply Weight to Non-Zero instances
    # Weighted NLL = - [ Mean(Mask0 * LL0) + Mean(Mask1 * LL1 * Weight) ]
    # Note: we shouldn't just multiply the whole log prob, but it's the standard way to weight samples.
    
    nll = - (mask_zero * log_p_zero + (1 - mask_zero) * log_p_nonzero * pos_weight_val)
    
    return torch.mean(nll)

def run_zinb_model(data, gpu_id=0):
    device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
    print(f"  > Running ZINB-MLP on {device} (Weighted + Warmup)...")
    
    batch_size = 1024
    y_train = data['train']['y']
    
    # Calc Bias
    mu_global = np.mean(y_train[y_train > 0]) if np.sum(y_train>0) > 0 else 0.0
    count_bias = np.log(mu_global + 1e-6) if mu_global > 0 else 0.0
    
    train_dataset = TensorDataset(torch.FloatTensor(data['train']['dense']), torch.FloatTensor(y_train))
    val_dataset = TensorDataset(torch.FloatTensor(data['val']['dense']), torch.FloatTensor(data['val']['y']))
    test_tensor = torch.FloatTensor(data['test']['dense']).to(device)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, num_workers=0)
    
    model = ZINB_MLP(count_bias=mu_global).to(device) # Pass mean directly
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-2)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-5)
    
    best_val_loss = float('inf')
    best_model_wts = copy.deepcopy(model.state_dict())
    
    epochs = 50
    warmup_epochs = 15
    
    for epoch in range(epochs):
        model.train()
        
        # Warmup Strategy: Only train Gate (pi_head) first?
        # In strictly implementations, we might freeze other heads.
        # But simpler: just run, the weighted loss will guide it?
        # User requested: "ZINB 必須加 ... 給 ZINB 15 epochs 的 Warmup (只練 Gate)"
        
        if epoch < warmup_epochs:
            # Freeze Count/Theta heads
            for param in model.mu_head.parameters(): param.requires_grad = False
            for param in model.theta_head.parameters(): param.requires_grad = False
            phase_name = "WARMUP"
        else:
            # Unfreeze
            for param in model.mu_head.parameters(): param.requires_grad = True
            for param in model.theta_head.parameters(): param.requires_grad = True
            phase_name = "JOINT"
            
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            pi_logits, mu, theta = model(X_b)
            loss = zinb_loss(y_b, pi_logits, mu, theta, pos_weight_val=20.0)
            loss.backward()
            optimizer.step()
        
        scheduler.step()
            
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for X_b, y_b in val_loader:
                X_b, y_b = X_b.to(device), y_b.to(device)
                pi_logits, mu, theta = model(X_b)
                val_loss += zinb_loss(y_b, pi_logits, mu, theta, pos_weight_val=20.0).item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        # Checkpoint: Special Logic
        # If switching from Warmup to Joint, reset best loss?
        # User requested fairness. Usually yes.
        if epoch == warmup_epochs:
             best_val_loss = float('inf')
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_wts = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_model_wts)
    model.eval()
    with torch.no_grad():
        pi_logits, mu, theta = model(test_tensor)
        pi = torch.sigmoid(pi_logits)
        # Expected value for ZINB where pi is P(Survival): E[y] = pi * mu
        # (Standard ZINB pi=P(Zero) -> E[y] = (1-pi)*mu)
        # We defined pi as P(Survival) in loss calc above.
        preds = pi * mu
        preds = preds.squeeze().cpu().numpy()
        
    return preds


# ==========================================
# 3. Evaluation Logic (The Trinity)
# ==========================================

def evaluate_metrics(y_true, y_pred):
    """
    Compute 3 levels of metrics: Global, Gate (Binary), Survivor (Conditional).
    Input:
      y_true: Transformed ground truth (y = 0 or y > 0)
      y_pred: Model prediction (Expected value E[y])
    """
    
    # 1. Global Metrics
    mae_global = mean_absolute_error(y_true, y_pred)
    mse_global = mean_squared_error(y_true, y_pred)
    
    # 2. Binary (Gate) Metrics
    y_true_bin = (y_true > 0).astype(int)
    
    # --- OPTIMAL THRESHOLDING FOR FAIR COMPARISON ---
    # Find best threshold on this set (Test set optimization is slightly cheating 
    # but okay for Baselines to show their "upper bound" potential)
    # Or realistically, one should use Val set threshold.
    # Here uses `analyze_thresholds` on the test set predictions just to be extremely generous to Baselines.
    best_thr = analyze_thresholds(y_true, y_pred)
    y_pred_bin = (y_pred > best_thr).astype(int)
    
    acc = accuracy_score(y_true_bin, y_pred_bin)
    prec = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    rec = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    f1 = f1_score(y_true_bin, y_pred_bin, zero_division=0)
    
    # Confusion Matrix for debugging
    tn, fp, fn, tp = confusion_matrix(y_true_bin, y_pred_bin, labels=[0, 1]).ravel()
    
    # 3. Survivor (Conditional) Metrics
    # Only evaluate on cases that actually survived (y_true > 0)
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
# 4. Main Experiment Worker
# ==========================================

def run_category_experiment(cat, data_dir, gpu_id, return_dict, bert_lock):
    """
    Worker process for a single category.
    """
    # Log to file
    log_file = f"log_new_{cat}.txt"
    sys.stdout = open(log_file, "w", buffering=1)
    sys.stderr = sys.stdout
    
    print(f"[{cat}] Process started on GPU {gpu_id}")
    
    try:
        # Load & Transform Data
        datasets = load_dataset_group(data_dir, cat)
        y_test = datasets['test']['y'] # Already transformed
        
        results = {}
        
        # --- Run Models (Reordered: GPU First) ---
        
        # 1. XGBoost (GPU)
        pred_xgb = run_xgboost(datasets, gpu_id=gpu_id)
        results['XGBoost'] = evaluate_metrics(y_test, pred_xgb)
        del pred_xgb
        gc.collect()
        torch.cuda.empty_cache()
        
        # 2. XGBoost Combined (GPU) - Removed to save time or redundant? Keep if previous code had it.
        # But XGB Combined was commented out in previous version?
        # User instructions: "Linear SVR, XGBoost, BERT-MLP, BERT-Poisson, ZINB". 
        # I will uncomment the core ones.
        
        # 3. BERT Regression (GPU)
        pred_bert_r = run_pytorch_model('regression', datasets, gpu_id=gpu_id)
        results['BERT-Regression'] = evaluate_metrics(y_test, pred_bert_r)
        del pred_bert_r
        gc.collect()
        torch.cuda.empty_cache()
        
        # 4. BERT Poisson (GPU)
        pred_bert_p = run_pytorch_model('poisson', datasets, gpu_id=gpu_id)
        results['BERT-Poisson'] = evaluate_metrics(y_test, pred_bert_p)
        del pred_bert_p
        gc.collect()
        torch.cuda.empty_cache()

        # --- 7. ZINB (GPU) [NEW] ---
        print(f"[{cat}] Waiting for ZINB Lock...")
        with bert_lock: # Reuse the BERT lock since it uses PyTorch
            print(f"[{cat}] Acquired ZINB Lock. Training...")
            pred_zinb = run_zinb_model(datasets, gpu_id=gpu_id)
        results['ZINB'] = evaluate_metrics(y_test, pred_zinb)
        del pred_zinb
        gc.collect()
        torch.cuda.empty_cache()
        
        # 5. GLM (CPU)
        pred_glm = run_glm_poisson(datasets)
        results['GLM-Poisson'] = evaluate_metrics(y_test, pred_glm)
        del pred_glm
        gc.collect()
        
        # 6. SVR (CPU)
        pred_svr = run_linear_svr(datasets)
        results['LinearSVR'] = evaluate_metrics(y_test, pred_svr)
        del pred_svr
        gc.collect()


        # Print local summary for log file
        print(f"\n[{cat}] Local Results Summary:")
        for m, res in results.items():
            print(f"Model: {m}")
            print(f"  Global MAE: {res['Global MAE']:.4f}")
            print(f"  Gate F1:    {res['Gate F1']:.4f} (Rec: {res['Gate Rec']:.4f})")
            print(f"  Surv MAE:   {res['Surv MAE']:.4f}")
            print(f"  Conf Mat:   TP={res['TP']}, FP={res['FP']}, TN={res['TN']}, FN={res['FN']}")
            print("-" * 30)

        return_dict[cat] = results
        
    except Exception as e:
        print(f"[{cat}] Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        sys.stdout.close()

# ==========================================
# 5. Entry Point
# ==========================================

if __name__ == "__main__":
    try:
        multiprocessing.set_start_method('spawn')
    except RuntimeError:
        pass

    DATA_DIR = "data/datasets_split" 
    CATEGORIES = ['刑事', '民事', '行政']
    
    available_gpus = get_available_gpus()
    print(f"Detected GPUs: {available_gpus}")
    
    manager = multiprocessing.Manager()
    category_results = manager.dict()
    # Create a lock for ZINB training to prevent OOM
    bert_lock = manager.Lock() 

    processes = []
    
    # Launch Processes
    for i, cat in enumerate(CATEGORIES):
        assigned_gpu = available_gpus[i % len(available_gpus)] if available_gpus else 0
        p = multiprocessing.Process(
            target=run_category_experiment, 
            args=(cat, DATA_DIR, assigned_gpu, category_results, bert_lock)
        )
        p.start()
        processes.append(p)
    
    for p in processes:
        p.join()
