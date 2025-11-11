import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os
from concurrent.futures import ThreadPoolExecutor

os.makedirs('data/raw', exist_ok=True)
print("Thư mục 'data/raw' đã được tạo hoặc tồn tại.")

base_url = 'http://quotes.toscrape.com/page/{}/'
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
}

def get_quotes_from_page(page_num):
    url = base_url.format(page_num)
    try:
        response = requests.get(url, headers=headers, timeout=10)
        
        # Xử lý khi trang không tồn tại (hết trang)
        if response.status_code == 404:
            print(f'Page {page_num}: Lỗi 404 - Hết trang.')
            return 'LAST_PAGE_REACHED' # Dùng signal để báo hết trang
        
        if response.status_code != 200:
            print(f'Page {page_num}: Lỗi HTTP {response.status_code}')
            return []
            
        try:
            soup = BeautifulSoup(response.text, 'lxml')
        except:
            soup = BeautifulSoup(response.text, 'html.parser')
            
        quotes = soup.find_all('div', class_='quote')
        if not quotes:
             print(f'Page {page_num}: Không tìm thấy quotes (có thể là trang cuối).')
             return 'LAST_PAGE_REACHED'
             
        quotes_data = []
        for quote in quotes:
            # Quote text (clean newlines/spaces)
            quote_text_elem = quote.find('span', class_='text')
            quote_text = quote_text_elem.text.strip() if quote_text_elem else 'N/A'
            quote_text = re.sub(r'\s+', ' ', quote_text.replace('\n', ' ').replace('\r', ' '))  # Clean all whitespace
            
            # Author
            author_elem = quote.find('small', class_='author')
            author = author_elem.text.strip() if author_elem else 'Unknown'
            
            # Tags
            tags_elem = quote.find('div', class_='tags')
            tags = [tag.text.strip() for tag in tags_elem.find_all('a', class_='tag')] if tags_elem else []
            tags_str = ', '.join(tags)
            
            # Length
            length = len(quote_text)
            
            # Rating from tags (cải tiến hơn bằng dict)
            rating_map = {
                'love': 5, 'inspirational': 4, 'life': 3, 
                'books': 2, 'sad': 1, 'world': 3, 'truth': 4
            }
            rating = 3 # Default
            for tag in tags:
                if tag.lower() in rating_map:
                    rating = rating_map[tag.lower()]
                    break # Lấy rating của tag đầu tiên có trong map

            quotes_data.append({
                'Quote': quote_text,
                'Author': author,
                'Tags': tags_str,
                'Length': length,
                'Rating': rating
            })
        
        print(f'Page {page_num}: {len(quotes)} quotes scraped.')
        time.sleep(0.5) # Giảm request rate
        return quotes_data
        
    except requests.exceptions.RequestException as e:
        print(f'Page {page_num}: Lỗi kết nối/timeout: {e}')
        return []

# --- Logic cào chính ---
MAX_PAGES = 100 # Cài đặt giới hạn an toàn
data = []
print("Bắt đầu scrape, tìm kiếm trang cuối...")

# Dùng vòng lặp thay vì ThreadPoolExecutor để kiểm tra signal LAST_PAGE_REACHED
for page in range(1, MAX_PAGES + 1):
    result = get_quotes_from_page(page)
    if result == 'LAST_PAGE_REACHED':
        print(f"Đã đạt đến trang cuối (trang {page-1}). Dừng cào.")
        break
    elif isinstance(result, list):
        data.extend(result)
        
    # Lưu tạm sau mỗi 10 trang (tăng robustness)
    if page % 10 == 0:
        pd.DataFrame(data).to_csv('data/raw/quotes_temp_backup.csv', index=False)
        print(f"--- Backup tạm thời sau trang {page} ---")

df = pd.DataFrame(data)
df = df[df['Length'] > 0]
df.to_csv('data/raw/quotes_raw.csv', index=False)
print(f'Tổng mẫu scraped: {len(df)}')
# In kết quả cho debug
print(df.head())
print(df.describe())
print(df['Author'].value_counts().head(5))