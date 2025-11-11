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
    response = requests.get(url, headers=headers)
    if response.status_code != 200:
        print(f'Lỗi page {page_num}: {response.status_code}')
        return []
    try:
        soup = BeautifulSoup(response.text, 'lxml')
    except:
        soup = BeautifulSoup(response.text, 'html.parser')
    print(f'Page {page_num}: Used parser')
    
    quotes = soup.find_all('div', class_='quote')
    print(f'Page {page_num}: {len(quotes)} quotes found')
    
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
        tags = [tag.text.strip() for tag in tags_elem.find_all('a', class_='tag') if tags_elem] if tags_elem else []
        tags_str = ', '.join(tags)
        
        # Length
        length = len(quote_text)
        
        # Rating from tags
        rating = 3
        if 'love' in tags_str.lower():
            rating = 5
        elif 'inspirational' in tags_str.lower():
            rating = 4
        elif 'life' in tags_str.lower():
            rating = 3
        elif 'books' in tags_str.lower():
            rating = 2
        elif 'sad' in tags_str.lower():
            rating = 1
        
        quotes_data.append({
            'Quote': quote_text,
            'Author': author,
            'Tags': tags_str,
            'Length': length,
            'Rating': rating
        })
    
    time.sleep(0.5)
    return quotes_data

# Full scrape 10 pages for 100 mẫu
num_pages = 10
data = []
print("Bắt đầu scrape 10 pages...")
with ThreadPoolExecutor(max_workers=4) as executor:
    futures = [executor.submit(get_quotes_from_page, page) for page in range(1, num_pages + 1)]
    for future in futures:
        try:
            data.extend(future.result())
        except Exception as e:
            print(f'Error in page: {e}')

df = pd.DataFrame(data)
df = df[df['Length'] > 0]
df.to_csv('data/raw/quotes_raw.csv', index=False)
print(f'Tổng mẫu scraped: {len(df)}')
print(df.head())
print(df.describe())
print(df['Author'].value_counts().head(10))
print(df['Tags'].value_counts().head(10))