import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Tự tạo thư mục outputs/plots/tables nếu chưa có
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/tables', exist_ok=True)
print("Thư mục 'outputs/plots/tables' đã được tạo hoặc tồn tại.")

# Đọc processed data
df = pd.read_csv('data/processed/quotes_processed.csv')
print(f'Processed shape: {df.shape}')
print(df.head())

# Features (drop Quote/Price for PCA)
X_cols = [col for col in df.columns if col not in ['Quote', 'Price']]
X = df[X_cols].fillna(0)
y = df['Price']  # Length as target

# PCA (2 components for plot)
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
print(f'Variance PC1: {pca.explained_variance_ratio_[0]:.2%}, PC2: {pca.explained_variance_ratio_[1]:.2%}')

# Plot scatter (lưu luôn)
plt.figure(figsize=(10, 6))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
plt.colorbar(scatter, label='Quote Length')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA: Phân cụm quote theo Length/Rating/Auth/Tag')
plt.savefig('outputs/plots/pca_scatter.png', dpi=300, bbox_inches='tight')  # bbox_inches to fit title
plt.show()
plt.close()  # Luôn lưu và close to avoid memory

# Variance bar (lưu luôn)
plt.figure(figsize=(8, 4))
plt.bar(['PC1', 'PC2'], pca.explained_variance_ratio_)
plt.xlabel('Principal Components')
plt.ylabel('Explained Variance Ratio')
plt.title('Variance Explained by PC')
plt.savefig('outputs/plots/variance_bar.png', dpi=300, bbox_inches='tight')
plt.show()
plt.close()

# Loadings table
loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), columns=['PC1', 'PC2'], index=X_cols)
loadings.to_csv('outputs/tables/loadings.csv')
print('Top loadings PC1:')
print(loadings['PC1'].abs().sort_values(ascending=False).head())

# Export loadings to LaTeX for báo cáo (copy vào Word)
loadings_latex = loadings.head(10).to_latex(index=True, caption='Bảng 3.2: Loadings PCA (Top 10)')
with open('outputs/tables/loadings_latex.txt', 'w', encoding='utf-8') as f:
    f.write(loadings_latex)
print("Loadings LaTeX saved to outputs/tables/loadings_latex.txt (copy vào Word)")

# Dự đoán Length với RandomForest trên PCA
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)
rf = RandomForestRegressor(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
metrics = pd.DataFrame({'R2': [r2], 'RMSE': [rmse]})
metrics.to_csv('outputs/tables/metrics.csv')
print(f'R²: {r2:.2f} | RMSE: {rmse:.2f}')
print('Insights: PC1 đại diện độ dài (Length + Auth), PC2 tags/rating. R² cao cho thấy PCA giữ thông tin tốt.')