import pandas as pd
import os
import glob
import numpy as np
import sys
from collections import Counter

def map_filename_to_label(filename):
    """
    Mapping Chinese filenames to English Labels & Sort Order.
    """
    # --- Civil (10-19) ---
    if "民事_continuous" in filename:
        return "Civil", "Aggregate", 10
    elif "清償借款" in filename or "清償債務" in filename:
        return "Civil", "Debt Collection", 11
    elif "損害賠償" in filename:
        return "Civil", "Tort / Damages", 12
    elif "分割共有物" in filename:
        return "Civil", "Partition of Prop.", 13
    elif "離婚" in filename:
        return "Civil", "Divorce", 14
        
    # --- Criminal (20-29) ---
    elif "刑事_continuous" in filename:
        return "Criminal", "Aggregate", 20
    elif "公共危險" in filename:
        return "Criminal", "Public Safety (DUI)", 21
    elif "竊盜" in filename or "詐欺" in filename:
        return "Criminal", "Fraud / Theft", 22
    elif "毒品" in filename:
        return "Criminal", "Drugs", 23
    elif "傷害" in filename:
        return "Criminal", "Injury", 24
        
    # --- Administrative (30-39) ---
    elif "行政_continuous" in filename:
        return "Administrative", "Aggregate", 30
    elif "交通裁決" in filename:
        return "Administrative", "Traffic Adj.", 31
    elif "稅" in filename:
        return "Administrative", "Tax Litigation", 32
    
    return "Unknown", filename, 99

def get_hazard_rates_extended(file_path):
    try:
        df = pd.read_csv(file_path)
        if 'disputability' not in df.columns:
            return None
        
        data = pd.to_numeric(df['disputability'], errors='coerce').dropna()
        data = data[data >= 1]
        
        if len(data) < 10:
            return None
            
        rates = []
        # --- [關鍵修改] 擴充到 k=4 ---
        for k in [1, 2, 3, 4]:
            n_at_risk = len(data[data >= k])
            n_stop = len(data[data == k])
            
            if n_at_risk > 0:
                rate = n_stop / n_at_risk
            else:
                rate = np.nan # 用 NaN 標記沒有數據的情況
            rates.append(rate)
            
        return rates
    except:
        return None

def format_color_cell(val, row_values):
    """
    Helper function to apply color based on row max/min.
    Handles NaN gracefully.
    """
    if pd.isna(val):
        return "-"
        
    # Filter out NaNs for min/max calculation
    valid_vals = [v for v in row_values if not pd.isna(v)]
    if not valid_vals:
        return "-"

    is_max = (val == max(valid_vals))
    is_min = (val == min(valid_vals))
    
    val_str = f"{val*100:.1f}\\%"
    
    if is_max:
        return f"\\textcolor{{red}}{{{val_str}}}"
    elif is_min:
        return f"\\textcolor{{blue}}{{{val_str}}}"
    else:
        return val_str

def generate_hazard_latex_4_cols():
    base_dir = "." 
    folders = [
        os.path.join(base_dir, 'data', 'datasets_csv')
    ]

    all_files = []
    for f in folders:
        if os.path.exists(f):
            files = glob.glob(os.path.join(f, '*.csv'))
            all_files.extend(files)
        else:
            print(f"% Warning: Folder not found: {f}", file=sys.stderr)

    results = []

    for file_path in sorted(all_files):
        file_name = os.path.basename(file_path)
        domain, case_type, sort_order = map_filename_to_label(file_name)
        
        if domain == "Unknown":
            continue

        rates = get_hazard_rates_extended(file_path)
        if rates:
            results.append({
                "domain": domain,
                "case_type": case_type,
                "r1": rates[0],
                "r2": rates[1],
                "r3": rates[2],
                "r4": rates[3], # 新增 r4
                "sort": sort_order
            })

    results.sort(key=lambda x: x['sort'])
    domain_counts = Counter(row['domain'] for row in results)

    print("% ==========================================")
    print("% Table 2: Empirical Hazard Rates (Up to Inst 4)")
    print("% ==========================================")
    print(r"\begin{table}[t]")
    print(r"\centering")
    # 更新 Caption: 強調 Inst 4 的角色
    print(r"\caption{Empirical Hazard Rates (Termination Probability). Data extended to Instance 4 reveals the ``Tail Dynamics'': while Criminal cases terminate abruptly, Civil/Administrative cases in the remand loop often exhibit persistent low termination rates even at the 4th instance. \textcolor{red}{\textbf{Red}}/\textcolor{blue}{\textbf{Blue}} indicates row max/min.}")
    print(r"\label{tab:hazard_rates}")
    print(r"\resizebox{\columnwidth}{!}{%")
    # 修改 column 定義，增加一個欄位
    print(r"\begin{tabular}{llrrrr}")
    print(r"\toprule")
    print(r"\textbf{Domain} & \textbf{Case Type} & \textbf{Inst. 1} & \textbf{Inst. 2} & \textbf{Inst. 3} & \textbf{Inst. 4} \\ \midrule")

    current_domain = None
    
    for row in results:
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

        # 顏色邏輯: 在 4 個數值中找最大最小
        row_vals = [row['r1'], row['r2'], row['r3'], row['r4']]
        
        r1_str = format_color_cell(row['r1'], row_vals)
        r2_str = format_color_cell(row['r2'], row_vals)
        r3_str = format_color_cell(row['r3'], row_vals)
        r4_str = format_color_cell(row['r4'], row_vals) # 新增
            
        print(f"{domain_str} & {c_type_str} & {r1_str} & {r2_str} & {r3_str} & {r4_str} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\end{table}")

if __name__ == "__main__":
    generate_hazard_latex_4_cols()