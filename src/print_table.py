import re
import pandas as pd
import os

def read_and_merge_logs():
    # Use path relative to this script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_path = os.path.dirname(current_dir) # Project Root
    
    files = [
        "log_new_民事.txt",
        "log_new_刑事.txt",
        "log_new_行政.txt",
        "log_SHN_民事.txt",
        "log_SHN_刑事.txt",
        "log_SHN_行政.txt"
    ]
    
    merged_content = ""
    for f in files:
        full_path = os.path.join(base_path, f)
        if os.path.exists(full_path):
            with open(full_path, "r", encoding="utf-8") as file:
                merged_content += file.read() + "\n------------------------------\n"
    return merged_content

# Read actual logs
full_log = read_and_merge_logs()

def parse_wide_table(log_text):
    data = []
    current_domain = "Civil"
    
    # Simple Parsing
    blocks = log_text.split("------------------------------")
    for block in blocks:
        if "民事" in block: current_domain = "Civil"
        if "刑事" in block: current_domain = "Crim."
        if "行政" in block: current_domain = "Admin."
        
        model_match = re.search(r"Model:\s*(.*?)\n|SHN Results Summary", block)
        if not model_match: continue
        model_name = model_match.group(1) if model_match.group(1) else "SHN (Ours)"
        if "SHN" in block and "Model:" not in block: model_name = "SHN (Ours)"
        
        metrics = re.search(r"Global MAE:\s*([\d\.]+).*?Gate F1:\s*([\d\.]+).*?Surv MAE:\s*([\d\.]+).*?Conf Mat:\s*TP=(\d+), FP=(\d+), TN=(\d+), FN=(\d+)", block, re.DOTALL)
        if metrics:
            g_mae, g_f1, s_mae, tp, fp, tn, fn = metrics.groups()
            tp, fp, fn = int(tp), int(fp), int(fn)
            rec = tp / (tp + fn) if (tp + fn) > 0 else 0
            
            data.append({
                "Domain": current_domain,
                "Model": model_name.strip(),
                "Global MAE": float(g_mae),
                "Gate F1": float(g_f1),
                "Recall": rec,
                "Surv MAE": float(s_mae)
            })
            
    df = pd.DataFrame(data)
    
    # Pivot Table
    pivot_gmae = df.pivot(index='Model', columns='Domain', values='Global MAE')
    pivot_gf1 = df.pivot(index='Model', columns='Domain', values='Gate F1')
    pivot_rec = df.pivot(index='Model', columns='Domain', values='Recall')
    pivot_surv = df.pivot(index='Model', columns='Domain', values='Surv MAE')
    
    # 計算 Average
    for p in [pivot_gmae, pivot_gf1, pivot_rec, pivot_surv]:
        p['Avg.'] = p.mean(axis=1)
    
    # 合併
    final_df = pd.concat([
        pivot_gmae.add_suffix('_GMAE'),
        pivot_gf1.add_suffix('_GF1'),
        pivot_rec.add_suffix('_Rec'),
        pivot_surv.add_suffix('_Surv')
    ], axis=1)
    
    # 確保 SHN 在最後
    final_df['IsSHN'] = final_df.index.to_series().apply(lambda x: 1 if "SHN" in x else 0)
    final_df = final_df.sort_values(by=['IsSHN', 'Avg._Rec'], ascending=[True, True])

    # 生成 LaTeX
    print(r"\begin{table*}[t]")
    print(r"\centering")
    print(r"\caption{Comparison of Global MAE, Gate F1, Recall, and Survivor MAE. \textbf{SHN} consistently outperforms baselines.}")
    print(r"\label{tab:main_results}")
    print(r"\resizebox{\textwidth}{!}{%")
    # 這裡很寬，可能需要調整
    print(r"\begin{tabular}{l|cccc|cccc|cccc|cccc}")
    print(r"\toprule")
    print(r"\multirow{2}{*}{\textbf{Model}} & \multicolumn{4}{c|}{\textbf{Global MAE} ($\downarrow$)} & \multicolumn{4}{c|}{\textbf{Gate F1} ($\uparrow$)} & \multicolumn{4}{c|}{\textbf{Gate Recall} ($\uparrow$)} & \multicolumn{4}{c}{\textbf{Survivor MAE} ($\downarrow$)} \\")
    print(r" & \textbf{Civ} & \textbf{Cri} & \textbf{Adm} & \textbf{Avg} & \textbf{Civ} & \textbf{Cri} & \textbf{Adm} & \textbf{Avg} & \textbf{Civ} & \textbf{Cri} & \textbf{Adm} & \textbf{Avg} & \textbf{Civ} & \textbf{Cri} & \textbf{Adm} & \textbf{Avg} \\ \midrule")
    
    # 找出最佳值
    best_gmae_avg = final_df['Avg._GMAE'].min()
    best_gf1_avg = final_df['Avg._GF1'].max()
    best_rec_avg = final_df['Avg._Rec'].max()
    best_surv_avg = final_df['Avg._Surv'].min()

    for i, (model_name, row) in enumerate(final_df.iterrows()):
        # Draw line before SHN if it's the last one
        if "SHN" in model_name and i > 0:
            print(r"\midrule")

        def fmt(val, is_best=False):
            if pd.isna(val): return "-"
            s = f"{val:.3f}"
            return f"\\textbf{{{s}}}" if is_best else s
            
        # Helper to get domain values
        def get_vals(metric_suffix, best_val, is_min=False):
            civ = fmt(row.get(f'Civil{metric_suffix}', 0))
            crim = fmt(row.get(f'Crim.{metric_suffix}', 0))
            admin = fmt(row.get(f'Admin.{metric_suffix}', 0))
            
            val_avg = row[f'Avg.{metric_suffix}']
            is_best = (val_avg == best_val)
            avg = fmt(val_avg, is_best=is_best)
            return f"{civ} & {crim} & {admin} & {avg}"

        s_gmae = get_vals('_GMAE', best_gmae_avg, is_min=True)
        s_gf1 = get_vals('_GF1', best_gf1_avg, is_min=False)
        s_rec = get_vals('_Rec', best_rec_avg, is_min=False)
        s_surv = get_vals('_Surv', best_surv_avg, is_min=True)
        
        print(f"{model_name} & {s_gmae} & {s_gf1} & {s_rec} & {s_surv} \\\\")

    print(r"\bottomrule")
    print(r"\end{tabular}%")
    print(r"}")
    print(r"\end{table*}")

parse_wide_table(full_log)