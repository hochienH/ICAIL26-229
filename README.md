# ICAIL26-229
The Source Codes and Datasets of the Paper "How Long Will Your Case Stay Alive? Survival Analysis Beyond Instances"
## Requirements

## Data Preprocessing

### First Part (For Empirical Studies)
This section describes how we preprocess the raw judgment data for the empirical studies. The source code is located in the `src` directory.

#### 1. Raw Data Extraction (From Judicial Yuan Open Data)
First, download the monthly judgment data (RAR files) from the Judicial Yuan Open Data platform and place them in the `data/raw_rar` directory.

Then, run the filter script to extract first-instance judgments:
```bash
python src/filter_cases.py
```
This script acts as a parser to:
1.  Read raw JSON files directly from RAR archives.
2.  Filter for **first-instance cases** (excluding appeals, rulings, etc.).
3.  Output flattened CSV files to `data/first_instance_case`.

#### 2. Dataset Generation (Pre-scraping)
Once the basic CSV files are ready in `data/first_instance_case`, run the preparation pipeline:
```bash
python src/prepare_data.py
```
This script performs the following steps:
1.  **Categorize**: Classifies judgments into specific case types (Civil, Criminal, Administrative) using `categorize_jtitle.py`.
2.  **Sample Time Series**: Selects cases for longitudinal analysis using `sample_time_series.py`.
3.  **Initialize Columns**: Adds the `disputability` column to the target CSV files (`add_column_disputability.py`).

#### 3. Disputability Data Acquisition (Scraping)
We use a Selenium-based scraper to query the "Case History" (歷審) from the Judicial Yuan's website and calculate the **Disputability Score**.

To run the scraper:
```bash
python src/disputability_scraper_multi.py
```
*   **Sequential Mode**: If you are running on a machine with limited resources or need to debug, change `MAX_WORKERS = 1` in `src/disputability_scraper_multi.py`.
*   **Parallel Mode**: By default, it runs with multiple workers to speed up data acquisition.
*   **Data Input**: The scraper looks for batch files in `data/server_batches`. You should split the files generated in step 1 into smaller batches for processing.

#### 3. Finalization (Post-scraping)
After obtaining the disputability scores, run the finalization script to merge results and create the clean datasets:
```bash
python src/finalize_data.py
```
This script will:
1.  **Update Files**: Map the scraped scores back to the original CSV files using `update_original_files.py`.
2.  **Create Final Datasets**: Filter out invalid entries and generate the final datasets in `data/datasets_csv` using `create_final_datasets.py`.

### Second Part (For Model Training)
### Dataset Links

## Empirical Studies

## Model Training
