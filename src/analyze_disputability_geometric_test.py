import pandas as pd
import os
import glob
import numpy as np
from scipy import stats

def analyze_geometric_fit():
    print("Analyze Geometric Distribution Fit (Goodness-of-Fit)")
    print("Hypothesis: Data follows Geometric(p) where P(X=k) = (1-p)^(k-1) * p for k >= 1")
    print("-" * 140)
    print(f"{'File':<35} | {'N':<6} | {'Mean':<6} | {'Est. p':<6} | {'Chi2 Stat':<10} | {'p-value':<10} | {'Result (alpha=0.05)':<20}")
    print("-" * 140)

    folders = [
        'data/datasets_csv'
    ]
    
    for folder in folders:
        if not os.path.exists(folder):
            continue
            
        files = sorted(glob.glob(os.path.join(folder, '*.csv')))
        
        for fpath in files:
            try:
                df = pd.read_csv(fpath)
                if 'disputability' not in df.columns:
                    continue
                
                # Preprocessing
                data = pd.to_numeric(df['disputability'], errors='coerce').dropna()
                data = data[data >= 1] # Geometric distribution defined for x >= 1
                
                N = len(data)
                if N < 10:
                    continue
                
                # 1. Estimate p (MLE for Geometric starting at 1 is 1/mean)
                mean_val = data.mean()
                p_hat = 1.0 / mean_val
                
                # 2. Create Bins for Chi-Square Test
                # We need to bin data such that expected freq is >= 5 typically.
                # Since geometric decays, we have counts for 1, 2, 3... 
                # We will group tails.
                
                max_val = int(data.max())
                observed_counts = data.value_counts().sort_index()
                
                # We'll construct bins dynamically
                # Bins: [1, 2, 3, ..., k+]
                # We perform the test.
                
                # Let's define reasonable fixed bins for comparison, but aggregate tail
                # Or just iterate 1..max and aggregate when expected < 5?
                # Simpler approach: Bins 1, 2, 3, 4, 5+ (since we saw max around 2-3 in means)
                
                # Actually, let's calculate observed and expected for k=1, 2, ...
                # until expected drops below 5, then sum the tail.
                
                obs_bins = []
                exp_bins = []
                
                current_k = 1
                cumulative_prob = 0.0
                
                while True:
                    # Expected Probability for X = k: (1-p)^(k-1) * p
                    prob_k = ((1 - p_hat) ** (current_k - 1)) * p_hat
                    expected_count = N * prob_k
                    
                    # If expected count is small (e.g. < 5) AND we have some bins already, 
                    # we should group this and ALL remaining probability into the "Tail" bin and stop.
                    # Ideally we want the tail bin to also have >= 5.
                    
                    # Logic:
                    # If expected_count < 5, we make this the last bin ">= current_k".
                    # Expected for ">= current_k" is N * (1 - cumulative_prob_so_far)
                    # BUT we need to check if that tail sum is big enough. 
                    # If not, we should have merged with previous.
                    
                    # Simplified logic for robust reporting:
                    # Bin 1, 2, 3, 4 ... up to K where expected < 5, then merge rest.
                    
                    if expected_count < 5:
                        # Stop here. This bin will be ">= current_k"
                        # But wait, if this tail is too small, we merge into previous.
                        # Let's calculate the tail probability mass.
                        prob_rest = 1.0 - cumulative_prob
                        exp_rest = N * prob_rest
                        
                        if exp_rest < 5 and len(obs_bins) > 0:
                            # Merge into previous bin
                            obs_bins[-1] += len(data[data >= current_k])
                            exp_bins[-1] += exp_rest
                        else:
                            # Create new tail bin
                            obs = len(data[data >= current_k])
                            obs_bins.append(obs)
                            exp_bins.append(exp_rest)
                        
                        break
                    
                    # Normal bin for X = current_k
                    obs = len(data[data == current_k])
                    obs_bins.append(obs)
                    exp_bins.append(expected_count)
                    
                    cumulative_prob += prob_k
                    current_k += 1
                
                # Degrees of Freedom
                # k bins. Constraints: sum(obs)=N (1), estimated p (1).
                # df = len(obs_bins) - 1 - 1 = len - 2
                
                df_dof = len(obs_bins) - 2
                
                if df_dof <= 0:
                    # Not enough bins to test (data too concentrated on 1)
                    res_str = "Not enough bins"
                    print(f"{os.path.basename(fpath):<35} | {N:<6} | {mean_val:<6.4f} | {p_hat:<6.4f} | {'-':<10} | {'-':<10} | {res_str:<20}")
                    continue

                chisq_stat, p_val = stats.chisquare(f_obs=obs_bins, f_exp=exp_bins, ddof=1) # ddof is for params estimated. stats.chisquare default df is k-1. We want k-1-p = k-2. So we remove 1 more df.
                # Note: stats.chisquare(..., ddof=delta) return chi2 with k - 1 - delta dof.
                # We want k - 2. So k - 1 - 1. ddof=1.
                
                res_str = "Reject H0 (Not Geom)" if p_val < 0.05 else "Fail to Reject (Fit)"
                
                # Format
                p_val_fmt = "< 0.0001" if p_val < 0.0001 else f"{p_val:.4f}"
                
                print(f"{os.path.basename(fpath):<35} | {N:<6} | {mean_val:<6.4f} | {p_hat:<6.4f} | {chisq_stat:<10.2f} | {p_val_fmt:<10} | {res_str:<20}")
                
            except Exception as e:
                print(f"Error {os.path.basename(fpath)}: {e}")

if __name__ == "__main__":
    analyze_geometric_fit()
