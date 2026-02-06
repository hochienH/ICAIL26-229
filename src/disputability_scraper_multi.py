import csv
import time
import os
import re
import glob
import multiprocessing
import logging
from concurrent.futures import ProcessPoolExecutor
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
# Adjust this to 1 if you do not have a powerful machine or GPU
# Set to > 1 for parallel processing (e.g. 8)
MAX_WORKERS = 4 

# -----------------------------------------------------------------------------
# Disputability Calculation Logic
# -----------------------------------------------------------------------------

def parse_case_info(text_line):
    # Regex to extract: Court, Year, Type, Number
    pattern = r"(.+?)\s+(\d+)\s*年度\s*(.+?)\s*字第\s*(\d+)\s*號"
    match = re.search(pattern, text_line)
    if match:
        return {
            "court": match.group(1).strip(),
            "year": match.group(2).strip(),
            "type": match.group(3).strip(),
            "number": match.group(4).strip()
        }
    return None

def calculate_disputability(jud_his_text, target_case_info):
    lines = jud_his_text.split('\n')
    unique_cases = set()
    
    t_year = target_case_info['year']
    t_type = target_case_info['type']
    t_number = target_case_info['number']

    for line in lines:
        parsed = parse_case_info(line)
        if parsed:
            case_key = (parsed['year'], parsed['type'], parsed['number'])
            unique_cases.add(case_key)
            
    disputability_score = 0
    relevant_cases = []

    for case in unique_cases:
        c_year, c_type, c_number = case
        
        # Rule: Target itself OR cases with '上' (Appeal) or '更' (Retrial)
        is_target = (c_year == t_year and c_type == t_type and c_number == t_number)
        has_shang = '上' in c_type
        has_geng = '更' in c_type
        
        if is_target or has_shang or has_geng:
            disputability_score += 1
            relevant_cases.append(case)
            
    return disputability_score, relevant_cases

# -----------------------------------------------------------------------------
# Worker Function
# -----------------------------------------------------------------------------

def process_single_csv(csv_path):
    """
    Worker process:
    1. Opens ONE browser instance.
    2. Processes all rows in the given CSV.
    3. Closes the browser.
    """
    pid = os.getpid()
    
    # Setup logging for this worker
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"worker_{os.path.basename(csv_path)}.log")
    
    logging.basicConfig(
        filename=log_file, 
        filemode='w',
        level=logging.INFO, 
        format='%(asctime)s - [%(levelname)s] - %(message)s',
        force=True
    )

    startup_delay = random.uniform(2, 10)
    logging.info(f"Worker {pid} waiting {startup_delay:.2f}s before starting...")
    time.sleep(startup_delay)
    
    logging.info(f"Starting task for {os.path.basename(csv_path)} (PID: {pid})")

    # Setup Selenium
    chrome_options = Options()
    
    # Critical flags for bypassing common detection or rendering issues in headless
    chrome_options.add_argument("--headless=new") 
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument("--disable-dev-shm-usage") 
    chrome_options.add_argument("--window-size=1920,1080")
    
    # Dynamic port assignment
    worker_port = 10000 + (pid % 20000)
    chrome_options.add_argument(f"--remote-debugging-port={worker_port}")
    chrome_options.add_argument("--remote-allow-origins=*")
    
    # Anti-detection features
    chrome_options.add_argument("--disable-blink-features=AutomationControlled")
    chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
    chrome_options.add_experimental_option('useAutomationExtension', False)
    chrome_options.add_argument("--ignore-certificate-errors")
    
    chrome_options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36")
    
    driver = None
    try:
        logging.info(f"Initializing ChromeDriver (PID: {pid}, Port: {worker_port})...")
        driver = webdriver.Chrome(options=chrome_options)
        
        # Anti-detection script
        try:
            driver.execute_cdp_cmd("Page.addScriptToEvaluateOnNewDocument", {
                "source": """
                    Object.defineProperty(navigator, 'webdriver', {
                        get: () => undefined
                    })
                """
            })
        except Exception as e:
            logging.warning(f"Failed to inject CDP script: {e}")

        logging.info("Driver initialized successfully.")
        
        # Read Data
        rows = []
        fieldnames = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            fieldnames = reader.fieldnames if reader.fieldnames else []
            if 'disputability' not in fieldnames:
                fieldnames.append('disputability')
            rows = list(reader)

        processed_count = 0
        
        for i, row in enumerate(rows):
            # Skip if already processed
            if 'disputability' in row and row['disputability'] and row['disputability'].strip():
                continue

            # Identify Filename -> ID
            filename_key = next((k for k in row.keys() if '檔名' in k), None)
            if not filename_key:
                continue
            
            filename_with_ext = row[filename_key]
            filename_id = filename_with_ext.replace('.json', '')
            
            url = f"https://judgment.judicial.gov.tw/FJUD/data.aspx?ty=JD&id={filename_id}"
            
            max_retries = 3
            retry_delay = 5 
            current_score = 0
            
            logging.info(f"Processing ID: {filename_id} | URL: {url}")

            for attempt in range(max_retries):
                try:
                    driver.get(url)
                    time.sleep(2) 
                    
                    error_in_title = "錯誤" in driver.title or "Error" in driver.title or "Access Denied" in driver.title
                    if error_in_title:
                        logging.warning(f"Page title indicates error: {driver.title}")
                        current_score = 0
                    else:
                        try:
                            jud_his_block = driver.find_element(By.ID, "JudHis")
                            jud_his_text = jud_his_block.text
                            
                            parts = filename_id.split(',')
                            if len(parts) >= 4:
                                target_info = {"year": parts[1], "type": parts[2], "number": parts[3]}
                                current_score, _ = calculate_disputability(jud_his_text, target_info)
                                
                                if current_score == 0:
                                    current_score = 1
                            else:
                                current_score = 0
                                
                        except Exception as e:
                            logging.warning(f"JudHis block not found: {e}")
                            current_score = 0
                    
                    break

                except Exception as e:
                    if attempt < max_retries - 1:
                        time.sleep(retry_delay)
                    else:
                        logging.error(f"Failed accessing {url}: {e}")
                        current_score = 0
            
            # Update row
            row['disputability'] = str(current_score)
            processed_count += 1
            
            # Write Back Immediately
            with open(csv_path, 'w', encoding='utf-8', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
                
    except Exception as e:
        logging.critical(f"CRITICAL ERROR processing {csv_path}: {e}", exc_info=True)
        print(f"[Worker {pid}] CRITICAL ERROR processing {csv_path}: {e}")
    finally:
        if driver:
            try:
                driver.quit()
            except:
                pass
    
    print(f"[Worker {pid}] Finished {os.path.basename(csv_path)}.")


# -----------------------------------------------------------------------------
# Main Manager
# -----------------------------------------------------------------------------

def main():
    import sys
    print("=== Disputability Scraper ===")
    print(f"MAX_WORKERS: {MAX_WORKERS}")

    # Determine batch directory
    # Expected location: ../data/server_batches (user should place split batches here)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    batch_dir = os.path.join(base_dir, 'data', 'server_batches')
    
    if not os.path.exists(batch_dir):
        print(f"Error: Batch directory not found at {batch_dir}")
        print("Please split your CSVs into batches and put them there.")
        return

    files_to_process = sorted(glob.glob(os.path.join(batch_dir, 'batch_*.csv')))
    
    if not files_to_process:
        print(f"No batch_*.csv files found in {batch_dir}.")
        return
        
    print(f"Found {len(files_to_process)} batch files to process.")
    
    # Processing Logic
    workers_to_use = min(len(files_to_process), MAX_WORKERS)
    
    if workers_to_use > 1:
        print(f"Starting ProcessPoolExecutor with {workers_to_use} workers...")
        with ProcessPoolExecutor(max_workers=workers_to_use) as executor:
            executor.map(process_single_csv, files_to_process)
    else:
        print("Running in sequential mode (Single Process)...")
        for f in files_to_process:
            process_single_csv(f)
        
    print("All tasks completed.")

if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
