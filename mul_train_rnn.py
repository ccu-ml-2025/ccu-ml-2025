import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# ===== 多任務模型定義 =====
class MultiTaskCNN(nn.Module):
    def __init__(self, num_classes_level=4, num_classes_years=3):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)

        self.fc_shared = nn.Linear(128, 128)
        self.fc_level = nn.Linear(128, num_classes_level)
        self.fc_years = nn.Linear(128, num_classes_years)

    def forward(self, x):
        # x = x.view(x.size(0), x.size(1), -1)  # (B, 27, 600)
        # x = x.transpose(1, 2)                # (B, 600, 27)
        x = x.permute(0, 3, 1, 2)  # (B, 6, 27, 100)
        x = x.reshape(x.size(0), 6, -1)  # (B, 6, 2700)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)         # (B, 128)
        x = torch.relu(self.fc_shared(x))
        return self.fc_level(x), self.fc_years(x)

# ===== Dataset 定義 =====
class MultiTaskWaveformDataset(Dataset):
    def __init__(self, data_dir, info_df, target_cols, segment_len=100):
        self.paths = []
        self.labels = []
        for file in os.listdir(data_dir):
            if file.endswith(".txt"):
                uid = int(file[:-4])
                row = info_df[info_df["unique_id"] == uid]
                if not row.empty:
                    self.paths.append(os.path.join(data_dir, file))
                    self.labels.append(tuple(row[c].values[0] for c in target_cols))
        self.segment_len = segment_len

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        with open(path) as f:
            lines = f.readlines()[1:]
            data = [list(map(int, l.strip().split())) for l in lines if l.strip() and len(l.strip().split()) == 6]
        data = np.array(data)
        total_len = len(data)
        segs = np.linspace(0, total_len, 28, dtype=int)
        segments = []
        for i in range(27):
            seg = data[segs[i]:segs[i+1]]
            if len(seg) >= self.segment_len:
                seg = seg[:self.segment_len]
            else:
                pad = np.zeros((self.segment_len - len(seg), 6))
                seg = np.vstack([seg, pad])
            segments.append(seg)
        tensor = torch.tensor(np.stack(segments), dtype=torch.float32)
        label1, label2 = self.labels[idx]
        return tensor, label1, label2

# ===== 主訓練流程 =====
def train_multitask(data_dir, info_csv, batch_size=32, epochs=15):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    df = pd.read_csv(info_csv)
    target_cols = ['level', 'play years']
    encoders = {col: LabelEncoder().fit(df[col]) for col in target_cols}
    for col in target_cols:
        df[col] = encoders[col].transform(df[col])

    players = df['mode'].unique()
    train_ids, test_ids = train_test_split(players, test_size=0.2, random_state=42)
    df_train = df[df['mode'].isin(train_ids)]
    df_test = df[df['mode'].isin(test_ids)]

    train_set = MultiTaskWaveformDataset(data_dir, df_train, target_cols)
    test_set = MultiTaskWaveformDataset(data_dir, df_test, target_cols)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = MultiTaskCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        for x, y1, y2 in train_loader:
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            out1, out2 = model(x)
            loss = criterion(out1, y1) + criterion(out2, y2)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        model.eval()
        preds1, preds2, labels1, labels2 = [], [], [], []
        with torch.no_grad():
            for x, y1, y2 in test_loader:
                x = x.to(device)
                out1, out2 = model(x)
                preds1.extend(torch.argmax(out1, dim=1).cpu().numpy())
                preds2.extend(torch.argmax(out2, dim=1).cpu().numpy())
                labels1.extend(y1.numpy())
                labels2.extend(y2.numpy())

        acc1 = accuracy_score(labels1, preds1)
        acc2 = accuracy_score(labels2, preds2)
        print(f"Epoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Level Acc: {acc1:.4f} | Years Acc: {acc2:.4f}")

    torch.save(model.state_dict(), "weight/multitask_model.pt")
    print("✅ 模型已儲存為 multitask_model.pt")

if __name__ == '__main__':
    train_multitask("train_data", "train_info.csv")
