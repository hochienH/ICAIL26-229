import pandas as pd
import os
import matplotlib.pyplot as plt

# --- 設定繪圖風格 ---
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False 

def plot_sparsity_with_inset_list():
    # Configuration
    datasets = [
        {'name': 'Criminal', 'path': 'data/datasets_csv/刑事_continuous.csv', 'thresh': 3, 'color': '#8e44ad'},
        {'name': 'Civil', 'path': 'data/datasets_csv/民事_continuous.csv', 'thresh': 3, 'color': '#2980b9'},
        {'name': 'Administrative', 'path': 'data/datasets_csv/行政_continuous.csv', 'thresh': 2, 'color': '#c0392b'}
    ]    # 翻譯表
    translation_map = {
        '貪污等': 'Corruption',
        '殺人等': 'Homicide',
        '違反森林法等': 'Forestry Act',
        '違反公司法等': 'Company Act',
        '違反槍砲彈藥刀械管制條例等': 'Firearms Act',
        '土地所有權移轉登記等': 'Land Transfer',
        '給付工程款等': 'Construct. Payment',
        '返還價金等': 'Return Purchase Price',
        '給付貨款等': 'Payment of Goods',
        '清償債務等': 'Debt Collection',
        '市地重劃': 'Urban Land Consol.',
        '追繳押標金': 'Bid Bond Recovery',
        '檢舉獎金': 'Whistleblower Bonus',
        '農地重劃': 'Farm Land Consol.',
        '監獄行刑法': 'Prison Act'
    }

    MIN_CASE_COUNT = 30
    
    # 使用 1x3 版面
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    
    for idx, ds in enumerate(datasets):
        path = ds['path']
        name = ds['name']
        threshold = ds['thresh']
        color = ds['color']
        ax = axes[idx]
        
        if not os.path.exists(path):
            continue
            
        try:
            df = pd.read_csv(path)
            
            # 清理與計算
            df['disputability'] = pd.to_numeric(df['disputability'], errors='coerce')
            df.dropna(subset=['disputability', 'JTITLE'], inplace=True)
            df['JTITLE'] = df['JTITLE'].astype(str).str.strip()
            
            df['is_high'] = df['disputability'] > threshold
            stats = df.groupby('JTITLE').agg(
                total_count=('disputability', 'count'),
                high_count=('is_high', 'sum')
            )
            
            valid_stats = stats[stats['total_count'] >= MIN_CASE_COUNT].copy()
            valid_stats['rate'] = (valid_stats['high_count'] / valid_stats['total_count']) * 100
            
            # 排序
            df_sorted = valid_stats.sort_values(by='rate', ascending=False).reset_index()
            
            if df_sorted.empty:
                continue

            # --- 1. 繪製乾淨的 Bar Chart (不加標籤) ---
            ax.bar(df_sorted.index, df_sorted['rate'], color=color, width=1.0, alpha=0.8)
            
            # --- 2. 在右上角建立 "Top 3 Contributors" 列表 ---
            # 準備文字內容
            top_text = "Top High-Risk Types:\n"
            for i in range(min(3, len(df_sorted))):
                row = df_sorted.iloc[i]
                raw_name = row['JTITLE']
                eng_name = translation_map.get(raw_name, raw_name)
                # 格式: 1. Corruption (28.0%)
                top_text += f"{i+1}. {eng_name} ({row['rate']:.1f}%)\n"
            
            # 加上 Sparsity 資訊
            zero_count = len(df_sorted[df_sorted['rate'] == 0])
            total_jtitles = len(df_sorted)
            sparsity_pct = (zero_count / total_jtitles) * 100
            sparsity_text = f"\nZero-Risk Types: {sparsity_pct:.1f}%"

            final_text = top_text + sparsity_text

            # 將文字方塊放在圖的內部 (右上)
            # transform=ax.transAxes 確保位置是相對座標 (0~1)
            ax.text(0.95, 0.95, final_text, 
                    transform=ax.transAxes, 
                    fontsize=11, 
                    verticalalignment='top', 
                    horizontalalignment='right',
                    bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.9, edgecolor='#cccccc'))

            # --- 設定標題與軸 ---
            ax.set_title(f"{name} Litigation\n(I > {threshold})", fontsize=14, fontweight='bold')
            ax.set_xlabel('Case Type Rank (Sorted)', fontsize=11)
            ax.set_xlim(-2, len(df_sorted)+5)
            
            ax.grid(axis='x', visible=False)
            ax.grid(axis='y', linestyle='--', alpha=0.5)

        except Exception as e:
            print(f"Error plotting {name}: {e}")

    axes[0].set_ylabel('Survivor Proportion (%)', fontsize=12)
    plt.tight_layout()
    
    # 存檔
    output_filename = 'hdr_sparsity_plot.pdf'
    plt.savefig(output_filename, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_filename}")

if __name__ == "__main__":
    plot_sparsity_with_inset_list()