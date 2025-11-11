import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
from scipy import stats

# Tự tạo thư mục data/processed nếu chưa có
os.makedirs('data/processed', exist_ok=True)
print("Thư mục 'data/processed' đã được tạo hoặc tồn tại.")

# Đọc raw data
df = pd.read_csv('data/raw/quotes_raw.csv')
print(f'Raw shape: {df.shape}')
print(df.head())

# Clean missing
df = df.dropna(subset=['Quote', 'Author'])
df['Rating'] = df['Rating'].fillna(df['Rating'].median())
df['Length'] = df['Length'].fillna(df['Length'].median())

# Outlier Z-score (giữ 95%)
z_scores = np.abs(stats.zscore(df[['Length', 'Rating']]))
df_clean = df[(z_scores < 3).all(axis=1)]
print(f'Sau outlier: {len(df_clean)} mẫu')

# One-hot giới hạn: top 5 Author and top 3 Tags (split Tags by ', ')
df['Tags'] = df['Tags'].str.split(', ')
df_exploded = df_clean.explode('Tags')
top_authors = df_clean['Author'].value_counts().head(5).index
top_tags = df_exploded['Tags'].value_counts().head(3).index
df_clean['Author'] = df_clean['Author'].apply(lambda x: x if x in top_authors else 'Other')
df_clean = pd.get_dummies(df_clean, columns=['Author'], prefix='Auth_')
df_clean['Tags'] = df_clean['Tags'].apply(lambda x: [t.strip() for t in x if t.strip() in top_tags] if isinstance(x, list) else [])
df_clean = pd.get_dummies(df_clean.explode('Tags'), columns=['Tags'], prefix='Tag_').groupby(level=0).sum()  # One-hot tags limited

# Features (ít cột: 2 numeric + 5 Auth + 3 Tag = 10 cột)
numeric_features = ['Length', 'Rating']
categorical_features = [col for col in df_clean.columns if col.startswith('Auth_') or col.startswith('Tag_')]
features = numeric_features + categorical_features[:8]  # Giới hạn 8 cột cat
X = df_clean[features].fillna(0)

# Scale
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', 'passthrough', categorical_features[:8])
    ]
)
X_scaled = preprocessor.fit_transform(X)

# Lưu processed
df_processed = pd.DataFrame(X_scaled, columns=features)
df_processed['Quote'] = df_clean['Quote'].values
df_processed['Price'] = df_clean['Length'].values  # Use Length as 'Price' for prediction
df_processed.to_csv('data/processed/quotes_processed.csv', index=False)
print(f'Processed shape: {X_scaled.shape}')
print(df_processed.head())
print('Số cột categorical: ', len([col for col in features if 'Auth_' in col or 'Tag_' in col]))