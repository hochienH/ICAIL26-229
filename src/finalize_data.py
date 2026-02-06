import os
import sys
import subprocess

def run_script(script_name):
    script_path = os.path.join(os.path.dirname(__file__), script_name)
    print(f"--- Running {script_name} ---")
    result = subprocess.run([sys.executable, script_path], check=False)
    if result.returncode != 0:
        print(f"Script {script_name} failed with code {result.returncode}")
        sys.exit(1)
    print(f"--- Finished {script_name} ---\n")

def main():
    print("=== Pipeline Stage 2: Finalization ===")
    
    # 1. Update original CSVs with scraped data
    run_script('update_original_files.py')
    
    # 2. Create final clean datasets
    run_script('create_final_datasets.py')
    
    print("=== Stage 2 Complete ===")
    print("Final datasets are available in 'data/datasets_csv'.")

if __name__ == "__main__":
    main()
