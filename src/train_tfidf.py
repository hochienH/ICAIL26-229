import os
import glob
import joblib
import jieba
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from concurrent.futures import ProcessPoolExecutor
import random
from pathlib import Path

# Import our custom utils
import judgement_utils

# Configuration
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
JUDGEMENTS_DIR = os.path.join(PROJECT_ROOT, "data/judgements")
OUTPUT_MODEL_PATH = os.path.join(PROJECT_ROOT, "models/tfidf_model.joblib")

# 抽樣參數
SAMPLE_SIZE = 100000  # 抽樣 10 萬個檔案來建立詞彙表
# 註：每個檔案有多個句子，所以總句子數可能會是一兩百萬，這對 TF-IDF 來說很足夠了且不會跑太久。

def get_sentences_from_file(file_path):
    """
    Worker function to extract tokenized sentences from a file
    """
    result = judgement_utils.process_single_file_content(file_path)
    if not result or not result['sentences']:
        return []
    
    # 對每個句子做斷詞
    tokenized_sentences = [judgement_utils.tokenize_text(s) for s in result['sentences']]
    return tokenized_sentences

def main():
    print("=== Training TF-IDF Model ===")
    
    # 建立模型目錄
    os.makedirs(os.path.dirname(OUTPUT_MODEL_PATH), exist_ok=True)
    
    # 1. 取得所有檔案列表
    print("Listing files...")
    all_files = list(Path(JUDGEMENTS_DIR).glob("*.json"))
    total_files = len(all_files)
    print(f"Total files found: {total_files}")
    
    # 2. 抽樣
    if total_files > SAMPLE_SIZE:
        print(f"Sampling {SAMPLE_SIZE} files for vocabulary building...")
        sampled_files = random.sample(all_files, SAMPLE_SIZE)
    else:
        sampled_files = all_files
        
    # 3. 收集語料 (Corpus Collection)
    print("Processing files to build corpus (Multi-process)...")
    corpus = []
    
    # 使用多進程加速讀取與斷詞
    # 注意：macOS 上 ProcessPoolExecutor 如果不設 max_workers，預設是 CPU 核心數
    with ProcessPoolExecutor(max_workers=8) as executor:
        results = list(tqdm(executor.map(get_sentences_from_file, sampled_files), total=len(sampled_files)))
        
    print("Flattening corpus...")
    # 將結果攤平 (List of Lists -> List of Strings)
    for res in results:
        corpus.extend(res)
        
    print(f"Total sentences for training: {len(corpus)}")
    
    # 4. 訓練 TF-IDF
    print("Fitting TfidfVectorizer...")
    # max_features: 限制詞彙量，避免矩陣過大。10萬詞對於法律文本通常足夠。
    # min_df: 去除極低頻詞
    vectorizer = TfidfVectorizer(max_features=100000, min_df=5, token_pattern=r'(?u)\b\w+\b')
    vectorizer.fit(corpus)
    
    print(f"Vocabulary size: {len(vectorizer.vocabulary_)}")
    
    # 5. 儲存模型
    joblib.dump(vectorizer, OUTPUT_MODEL_PATH)
    print(f"Model saved to {OUTPUT_MODEL_PATH}")

if __name__ == "__main__":
    import multiprocessing
    # Mac/Linux sometimes needs this for jieba/sklearn in MP
    multiprocessing.set_start_method('fork', force=True) 
    main()
