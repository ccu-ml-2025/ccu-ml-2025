import os
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.cluster import KMeans, DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# ===== 參數設定 =====
data_dir = "tabular_data_train2"
info_path = "train_info.csv"
target_col = "play years"  # 可選: 'level' / 'play years'
n_clusters = 3        # 與實際類別數對齊

# ===== 讀取資料與標籤 =====
info = pd.read_csv(info_path)
data_list = []
labels = []

for file in Path(data_dir).glob("*.csv"):
    uid = int(file.stem)
    row = info[info["unique_id"] == uid]
    if row.empty:
        continue
    label = row[target_col].values[0]
    df = pd.read_csv(file)
    vec = df.mean(axis=0).values  # 取特徵的平均向量 (34 維)
    data_list.append(vec)
    labels.append(label)

X = np.array(data_list)
y = np.array(labels)

# ===== 標準化 + 降維 =====
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 可選: PCA / t-SNE
X_embedded = TSNE(n_components=2, random_state=42).fit_transform(X_scaled)

# ===== 聚類分析 =====
kmeans = KMeans(n_clusters=n_clusters, random_state=42)
preds = kmeans.fit_predict(X_scaled)

# ===== 繪圖：2D 嵌入 + 顏色顯示聚類分群 =====
plt.figure(figsize=(8, 6))
plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=preds, cmap='tab10', s=20)
plt.title(f"KMeans Clustering Visualization ({target_col})")
plt.xlabel("Dim 1")
plt.ylabel("Dim 2")
plt.colorbar(label="Cluster")
plt.tight_layout()
plt.savefig(f"clustering_{target_col}.png")
print(f"✅ 已儲存聚類圖: clustering_{target_col}.png")

# ===== 混淆矩陣 (非必然對應但可檢查一致性) =====
cm = confusion_matrix(y, preds)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Cluster vs True Label Confusion Matrix")
plt.xlabel("Cluster ID")
plt.ylabel(f"True {target_col}")
plt.tight_layout()
plt.savefig(f"confusion_{target_col}.png")
print(f"✅ 已儲存混淆矩陣圖: confusion_{target_col}.png")
