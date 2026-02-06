import re
import json
import logging
import jieba

# 設定 jieba 字典（如果有自定義字典可以在這裡加載）
# jieba.set_dictionary('path/to/dict.txt')

def extract_main_text(jfull_content):
    """
    從 JFULL 內容中提取主文部分。
    邏輯：
    1. 嘗試用 Regex 抓取 "主文" ... "中華民國" 之間的內容。
    2. 若失敗，返回 False 及 原始內容 (Fallback)。
    
    Returns:
        tuple: (is_extracted, text)
        - is_extracted (bool): 是否成功經由正則提取
        - text (str): 提取出的文字 或 原始文字
    """
    if not jfull_content:
        return False, ""

    # 定義要匹配的正則表達式模式
    pattern1 = r'^\s*主\s*文\s*$'  # 匹配 "主文"
    pattern2 = r'^\s*中\s*華\s*民\s*國.*年.*月.*日\s*$'  # 匹配 "中華民國...年...月...日"
    
    # 編譯正則表達式，使用 MULTILINE 模式
    regex1 = re.compile(pattern1, re.MULTILINE)
    regex2 = re.compile(pattern2, re.MULTILINE)
    
    # 找到主文的位置
    main_match = regex1.search(jfull_content)
    
    # 找到日期的位置 (找最後一個日期，通常在結尾)
    # 不過判決書結構通常是 主文 -> 事實 -> 理由 -> 日期
    # 有時候 regex2 會匹配到內文引用的日期。
    # 為了保險，我們先簡單搜尋。若要嚴謹可能要找最後一個 match。
    date_matches = list(regex2.finditer(jfull_content))
    
    if main_match and date_matches:
        # 取最後一個日期做為結尾 (通常是判決結尾日期)
        # 但有風險是如果附錄有日期。
        # 安全起見，取 main_match 之後的第一個日期？
        # 一般判決書： 主文 ... (一段文字) ... 中華民國XX年
        
        # 尋找 main_match 之後的第一個日期
        target_date_match = None
        for dm in date_matches:
            if dm.start() > main_match.end():
                target_date_match = dm
                break
        
        if target_date_match:
            # 提取主文部分
            main_text = jfull_content[main_match.end():target_date_match.start()]
            return True, main_text.strip()

    # Fallback: 沒抓到，回傳全部
    return False, jfull_content

def clean_and_split_text(text):
    """
    清理文字並按句號分割成句子
    
    Args:
        text (str): 原始文字
        
    Returns:
        list: 句子列表
    """
    if not text:
        return []
    
    # 移除多餘空白 (但在中文不用移除所有空白，有時候排版需要)
    # 這裡選擇移除換行和連續空白，變成緊湊字串
    cleaned_text = re.sub(r'\s+', '', text)
    
    # 按句號分割
    # 可以考慮加入分號或其他終止符？目前需求是「句號」
    parts = cleaned_text.split('。')
    
    # 過濾與清理
    sentences = []
    for p in parts:
        p = p.strip()
        # 過濾太短的句子 (比如 "一"、"二" 這種標號，或是純標點)
        if len(p) > 1: 
            sentences.append(p)
            
    return sentences

def tokenize_text(text):
    """
    使用 Jieba 進行斷詞，供 TF-IDF 使用
    """
    return " ".join(jieba.cut(text))

def process_single_file_content(file_path):
    """
    讀取並解析單個 JSON 檔案
    Returns:
        dict: {
            'filename': str,
            'sentences': list[str],
            'extraction_method': str ('regex' or 'fallback')
        }
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        jfull = data.get('JFULL', '')
        
        is_extracted, text_content = extract_main_text(jfull)
        sentences = clean_and_split_text(text_content)
        
        return {
            'filename': file_path.name if hasattr(file_path, 'name') else str(file_path),
            'sentences': sentences,
            'extraction_method': 'regex' if is_extracted else 'fallback'
        }
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None
