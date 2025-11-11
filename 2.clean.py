import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
import numpy as np
from scipy import stats

# Tự tạo thư mục data/processed nếu chưa có
os.makedirs('data/processed', exist_ok=True)

# Đọc raw data
df = pd.read_csv('data/raw/quotes_raw.csv')

# Clean missing
df = df.dropna(subset=['Quote', 'Author'])
df['Rating'] = df['Rating'].fillna(df['Rating'].median())
df['Length'] = df['Length'].fillna(df['Length'].median())

# Xử lý Outlier Z-score (giữ 95%)
z_scores = np.abs(stats.zscore(df[['Length', 'Rating']]))
df_clean = df[(z_scores < 3).all(axis=1)].copy()
print(f'Sau xử lý outlier: {len(df_clean)} mẫu')

# Biến đổi Log cho Length nếu phân phối bị lệch
# Thêm 1 (+1) để xử lý giá trị Length=0 nếu có (mặc dù đã drop)
df_clean['Log_Length'] = np.log1p(df_clean['Length'])

# One-hot: top 10 Author and top 5 Tags
df_clean['Tags_list'] = df_clean['Tags'].str.split(', ') # Tạo cột list để explode
df_exploded = df_clean.explode('Tags_list')
top_authors = df_clean['Author'].value_counts().head(10).index
top_tags = df_exploded['Tags_list'].value_counts().head(5).index

# Gán 'Other' cho Author không nằm trong Top 10
df_clean['Author_Grouped'] = df_clean['Author'].apply(lambda x: x if x in top_authors else 'Other')
df_processed = pd.get_dummies(df_clean, columns=['Author_Grouped'], prefix='Auth_')

# One-hot tags giới hạn (chỉ giữ tags thuộc top 5)
df_processed['Tags_list'] = df_processed['Tags_list'].apply(lambda x: [t.strip() for t in x if t.strip() in top_tags] if isinstance(x, list) else [])
# Sử dụng explode và groupby để tạo cột one-hot cho Tags
df_processed = pd.get_dummies(df_processed.explode('Tags_list'), columns=['Tags_list'], prefix='Tag_').groupby(level=0).sum()

# --- Định nghĩa Features cho PCA và Mô hình ---
numeric_features = ['Log_Length', 'Rating']
# Lấy tất cả cột One-hot mới tạo
categorical_features = [col for col in df_processed.columns if col.startswith('Auth_') or col.startswith('Tag_')]
features_for_pca = numeric_features + categorical_features

# Tạo X_raw (dữ liệu thô đã qua one-hot/log) và X (cho scaling)
X_raw = df_processed[features_for_pca].fillna(0).copy()
X = X_raw.copy()

# Scale các biến số
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', 'passthrough', [col for col in X.columns if col not in numeric_features]) # Cat features giữ nguyên
    ],
    remainder='passthrough'
)
X_scaled = preprocessor.fit_transform(X)

# Lưu processed (Dữ liệu đã Scale)
df_scaled = pd.DataFrame(X_scaled, columns=features_for_pca, index=df_processed.index)
df_processed_final = pd.concat([df_scaled, df_processed[['Quote', 'Length']]], axis=1) # Giữ Length gốc làm target

df_processed_final.to_csv('data/processed/quotes_processed.csv', index=False)
X_raw.to_csv('data/processed/quotes_raw_features.csv', index=False) # Lưu X_raw để so sánh mô hình
print(f'Processed shape (Scaled): {df_processed_final.shape}')
print(f'Số lượng features cho PCA: {len(features_for_pca)}')
print(df_processed_final.head())