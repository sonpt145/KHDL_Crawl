import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_squared_error
import numpy as np

# Tự tạo thư mục outputs
os.makedirs('outputs/plots', exist_ok=True)
os.makedirs('outputs/tables', exist_ok=True)

# Đọc processed data (Scaled) và Raw Features (Không Scale/Log)
df_scaled = pd.read_csv('data/processed/quotes_processed.csv')
X_raw_features = pd.read_csv('data/processed/quotes_raw_features.csv') # Dữ liệu trước khi Scale

# Features và Target
X_cols = [col for col in df_scaled.columns if col not in ['Quote', 'Length']]
X_scaled = df_scaled[X_cols].fillna(0)
y = df_scaled['Length']  # Length gốc làm target

# --- 3.1. Xác định số lượng Components (Scree Plot) ---
pca_full = PCA(n_components=None)
pca_full.fit(X_scaled)
explained_variance_ratio = pca_full.explained_variance_ratio_

# Vẽ Scree Plot và Cumulative Explained Variance
plt.figure(figsize=(10, 5))

# Biểu đồ thanh (Explained Variance Ratio)
plt.subplot(1, 2, 1)
plt.bar(range(1, len(explained_variance_ratio) + 1), explained_variance_ratio)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Explained Variance Ratio')

# Biểu đồ đường (Cumulative Explained Variance)
plt.subplot(1, 2, 2)
cumulative_variance = np.cumsum(explained_variance_ratio)
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
# Tìm k components giữ 90% variance
k_90 = np.argmax(cumulative_variance >= 0.9) + 1
plt.axhline(0.9, color='red', linestyle='--', label='90% Variance')
plt.axvline(k_90, color='green', linestyle='--', label=f'k={k_90} (90%)')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.legend()
plt.tight_layout()
plt.savefig('outputs/plots/pca_scree_plot.png', dpi=300)
plt.show()
plt.close()

print(f"Số components giữ 90% Variance: k={k_90}")
# Chọn k_optimal để dùng trong PCA chính (ví dụ 3 hoặc 4 để có sự ổn định hơn 2)
k_optimal = k_90 if k_90 > 2 else 4 
if k_optimal > len(X_cols): k_optimal = len(X_cols) # Đảm bảo k không vượt quá số features

# --- 3.2. PCA với k_optimal components ---
pca = PCA(n_components=k_optimal)
X_pca = pca.fit_transform(X_scaled)
print(f'PCA: Dùng k={k_optimal} components (Giải thích {cumulative_variance[k_optimal-1]:.2%} Variance)')

# Plot scatter (Chỉ dùng 2 components đầu để trực quan hóa)
plt.figure(figsize=(10, 6))
pca_2d = PCA(n_components=2).fit_transform(X_scaled) # Chỉ để vẽ
scatter = plt.scatter(pca_2d[:, 0], pca_2d[:, 1], c=y, cmap='viridis', s=50, alpha=0.7)
plt.colorbar(scatter, label='Quote Length (Target)')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.title('PCA 2D: Phân cụm Quote')
plt.savefig('outputs/plots/pca_scatter_2d.png', dpi=300, bbox_inches='tight')
plt.close()

# Loadings table (Dùng k_optimal components)
pc_names = [f'PC{i+1}' for i in range(k_optimal)]
loadings = pd.DataFrame(pca.components_.T * np.sqrt(pca.explained_variance_), 
                        columns=pc_names, 
                        index=X_cols)
loadings.to_csv('outputs/tables/loadings_k_optimal.csv')
print('\nTop 5 Loadings PC1:')
print(loadings['PC1'].abs().sort_values(ascending=False).head())

# Export loadings to LaTeX
loadings_latex = loadings.head(10).to_latex(index=True, caption=f'Bảng 3.2: Loadings PCA (Top 10, k={k_optimal})')
with open('outputs/tables/loadings_latex.txt', 'w', encoding='utf-8') as f:
    f.write(loadings_latex)

# --- 3.3. So sánh Dự đoán Length (Random Forest) ---

def train_and_evaluate_rf(X_data, y_data, name):
    """Huấn luyện và đánh giá RF Regressor."""
    X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred = rf.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    return {'Model': name, 'R2': r2, 'RMSE': rmse}

# 1. Mô hình trên Dữ liệu Gốc (X_raw_features)
metrics_raw = train_and_evaluate_rf(X_raw_features, y, 'Raw_Features')

# 2. Mô hình trên Dữ liệu sau PCA (X_pca)
metrics_pca = train_and_evaluate_rf(pd.DataFrame(X_pca), y, f'PCA_k{k_optimal}')

# Tổng hợp và lưu kết quả
metrics_df = pd.DataFrame([metrics_raw, metrics_pca])
metrics_df.to_csv('outputs/tables/metrics_comparison.csv', index=False)
print('\n--- So sánh Hiệu suất Dự đoán Length (Random Forest) ---')
print(metrics_df.to_markdown(index=False))

# Insights cuối cùng
print('\nInsights: So sánh R² cho thấy PCA giữ thông tin rất tốt (R² tương đương/hơi giảm nhẹ), nhưng đã giảm chiều từ '
      f'{len(X_cols)} features xuống còn {k_optimal} features.')