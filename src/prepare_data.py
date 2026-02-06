import os
import sys
import subprocess

def run_script(script_name):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"--- Running {script_name} ---")
    result = subprocess.run([sys.executable, script_path], check=False)
    if result.returncode != 0:
        print(f"Script {script_name} failed with code {result.returncode}")
        # Not raising, allowing pipeline to proceed or partial manual intervention
    print(f"--- Finished {script_name} ---\n")

def main():
    print("=== Pipeline Stage 1: Preparation ===")
    
    # 1. (Optional) Filter Cases from Raw RARs
    # run_script('filter_cases.py')
    
    # 2. Categorize
    run_script('categorize_jtitle.py')
    
    # 3. Sample Time Series
    run_script('sample_time_series.py')
    
    # 4. Add Columns
    run_script('add_column_disputability.py')
    
    print("=== Stage 1 Complete ===")
    print("Next steps:")
    print("1. Split the generated CSVs in 'data/categorized_case' and 'data/sampled_time_series'")
    print("   into batches (e.g., 20 files per batch) and place them in 'data/server_batches'.")
    print("2. Run 'python src/disputability_scraper_multi.py' to acquire data.")

if __name__ == "__main__":
    main()
