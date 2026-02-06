import pandas as pd
import os
import glob
import numpy as np
import sys
from collections import Counter

def map_filename_to_label(filename):
    """
    將中文檔名映射到論文使用的英文標籤。
    回傳: (Domain, Case Type, Sort Order)
    """
    # --- Civil (10-19) ---
    if "民事_continuous" in filename: return "Civil", "Aggregate", 10
    elif "清償借款" in filename: return "Civil", "Debt Collection", 11
    elif "損害賠償" in filename: return "Civil", "Tort / Damages", 12
    elif "分割共有物" in filename: return "Civil", "Partition of Prop.", 13
    elif "離婚" in filename: return "Civil", "Divorce", 14
    # --- Criminal (20-29) ---
    elif "刑事_continuous" in filename: return "Criminal", "Aggregate", 20
    elif "公共危險" in filename: return "Criminal", "Public Safety (DUI)", 21
    elif "竊盜" in filename: return "Criminal", "Fraud / Theft", 22
    elif "毒品" in filename: return "Criminal", "Drugs", 23
    elif "傷害" in filename: return "Criminal", "Injury", 24
    # --- Administrative (30-39) ---
    elif "行政_continuous" in filename: return "Administrative", "Aggregate", 30
    elif "交通裁決" in filename: return "Administrative", "Traffic Adj.", 31
    elif "稅" in filename: return "Administrative", "Tax Litigation", 32
    
    return "Unknown", filename, 99

def calculate_comparative_stats(file_path, domain):
    """
    計算 Overall 與 Survivor 的統計數據
    [修正] 根據 domain 動態調整 Survivor 的定義門檻
    """
    try:
        df = pd.read_csv(file_path)
        if 'disputability' not in df.columns: return None
        data = pd.to_numeric(df['disputability'], errors='coerce').dropna()
        
        # 1. Overall Stats (I >= 1)
        valid_data = data[data >= 1]
        if len(valid_data) < 10: return None
        
        mean_all = valid_data.mean()
        var_all = valid_data.var()
        d_all = var_all / mean_all if mean_all > 0 else np.nan
        
        # 2. Survivor Stats (Domain-Specific Threshold)
        # 行政訴訟: Survivor I > 2 (即 >= 3)
        # 民事刑事: Survivor I > 3 (即 >= 4)
        if domain == "Administrative":
            tail_data = data[data > 2]
        else:
            tail_data = data[data > 3]
        
        
        var_tail = tail_data.var()
        mean_tail = tail_data.mean()
        d_tail = var_tail / mean_tail if mean_tail > 0 else np.nan
            
        return {
            "n": len(valid_data),
            "mean_all": mean_all,
            "d_all": d_all,
            "d_tail": d_tail
        }
    except:
        return None

def generate_new_table1():
    base_dir = "." # 請確認路徑
    folders = [os.path.join(base_dir, 'data', 'datasets_csv')]

    all_files = []
    for f in folders:
        if os.path.exists(f):
            files = glob.glob(os.path.join(f, '*.csv'))
            all_files.extend(files)

    stats_results = []

    # --- 資料處理 ---
    for file_path in sorted(all_files):
        file_name = os.path.basename(file_path)
        domain, case_type, sort_order = map_filename_to_label(file_name)
        
        if domain == "Unknown": continue

        # [修正] 傳入 domain 參數
        stats = calculate_comparative_stats(file_path, domain)
        if stats:
            stats_results.append({
                "domain": domain,
                "case_type": case_type,
                "sort": sort_order,
                **stats
            })

    # --- 排序 ---
    stats_results.sort(key=lambda x: x['sort'])
    
    # 計算每個 Domain 的行數
    domain_counts = Counter(row['domain'] for row in stats_results)

    # --- 生成 LaTeX ---
    print("% ==========================================")
    print("% Table 1: Comparative Statistics (Corrected Thresholds)")
    print("% ==========================================")
    print(r"\begin{table}[t]")
    print(r"\centering")
    # Caption 更新說明：分別定義了 threshold
    print(r"\caption{Litigation Intensity Statistics. The comparison between \textbf{Overall D} ($I \ge 1$) and \textbf{Survivor D} ($I > 3$ for Civil/Crim., $I > 2$ for Admin.) reveals the dual-phase dynamics: extreme underdispersion vanishes in the tail, yet the process remains structured ($D < 1$). \textit{Note: Sample size $N=20,000$ for specific types, $N=100,000$ for aggregates.}}")
    print(r"\label{tab:descriptive_stats}")
    print(r"\resizebox{\columnwidth}{!}{%")
    print(r"\begin{tabular}{llrrr}")
    print(r"\toprule")
    # 雙層表頭
    print(r"& & \textbf{Overall} & \textbf{Overall} & \textbf{Survivor} \\")
    # 修改表頭顯示：簡單寫 Survivor MVR 即可，Caption 已解釋定義
    print(r"\textbf{Domain} & \textbf{Case Type} & \textbf{Mean} & \textbf{D} & \textbf{D} \\ \midrule")

    current_domain = None
    
    for row in stats_results:
        if current_domain is not None and row['domain'] != current_domain:
            print(r"\midrule")
        
        if row['domain'] != current_domain:
            count = domain_counts[row['domain']]
            domain_str = f"\\multirow{{{count}}}{{*}}{{{row['domain']}}}"
            current_domain = row['domain']
        else:
            domain_str = ""

        c_type_str = row['case_type']
        if "Baseline" in c_type_str:
            c_type_str = f"\\textit{{{c_type_str}}}"

        mean_str = f"{row['mean_all']:.2f}"
        
        d_all_str = f"{row['d_all']:.2f}"
            
        if pd.isna(row['d_tail']):
            d_tail_str = "-"
        else:
            d_tail_str = f"{row['d_tail']:.2f}"

        print(f"{domain_str} & {c_type_str} & {mean_str} & {d_all_str} & {d_tail_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\end{table}")

if __name__ == "__main__":
    generate_new_table1()