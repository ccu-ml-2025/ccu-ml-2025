import os
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from collections import Counter
import numpy as np

# ===== Mixup 增強 =====
def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

# ===== 資料增強 =====
def add_gaussian_noise(tensor, mean=0.0, std=0.05):
    noise = torch.randn_like(tensor) * std + mean
    return tensor + noise

def random_augment(tensor):
    if torch.rand(1) < 0.5:
        tensor = add_gaussian_noise(tensor)
    return tensor


# ===== Dataset 定義 =====
class WaveformDataset(Dataset):
    def __init__(self, data_dir, label_dict, num_segments=27, target_length=100, augment_fn=None):
        self.file_paths = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.txt')]
        self.file_paths = [f for f in self.file_paths if os.path.splitext(os.path.basename(f))[0] in label_dict]
        self.label_dict = label_dict
        self.num_segments = num_segments
        self.target_length = target_length
        self.augment_fn = augment_fn  # 可為 None

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        uid = os.path.splitext(os.path.basename(path))[0]
        label = self.label_dict[uid]
        tensor = self.read_waveform_txt(path)
        if self.augment_fn:
            tensor = self.augment_fn(tensor)
        return tensor, label

    def read_waveform_txt(self, file_path):
        with open(file_path) as f:
            lines = f.readlines()[1:]
            data = [list(map(int, line.strip().split())) for line in lines if line.strip() and len(line.strip().split()) == 6]
        data = np.array(data)
        total_len = len(data)
        segment_idx = np.linspace(0, total_len, self.num_segments + 1, dtype=int)
        segments = []
        for i in range(self.num_segments):
            seg = data[segment_idx[i]:segment_idx[i + 1]]
            if len(seg) >= self.target_length:
                seg = seg[:self.target_length]
            else:
                pad = np.zeros((self.target_length - len(seg), 6))
                seg = np.vstack([seg, pad])
            segments.append(seg)
        result = np.stack(segments)
        return torch.tensor(result, dtype=torch.float32)  # shape: (num_segments, target_length, 6)

# ===== 模型定義 =====
class CNN1DClassifier(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv1d(6, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(64)
        self.conv2 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # x: (batch, 27, 100, 6)
        x = x.view(x.size(0), x.size(1), -1)  # -> (batch, 27, 600)
        x = x.transpose(1, 2)  # -> (batch, 600, 27)
        x = x.transpose(1, 2)  # -> (batch, 27, 600)
        x = x.transpose(1, 2)  # -> (batch, 600, 27)
        x = x[:, :6, :]  # just keep 6 channels (reshape bug fix)
        x = torch.relu(self.bn1(self.conv1(x)))
        x = torch.relu(self.bn2(self.conv2(x)))
        x = self.dropout(x)
        x = self.pool(x).squeeze(-1)
        return self.fc(x)

# ===== 主訓練流程 =====
def train_rnn(data_dir, info_csv, task, num_classes, batch_size=32, num_epochs=15, group="player_id", use_cuda=True, augment=False):
    device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")
    augment_fn = random_augment if augment else None

    df = pd.read_csv(info_csv)
    unique_players = df[group].unique()
    train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)

    train_dict, test_dict = {}, {}
    le = LabelEncoder()
    df[task] = le.fit_transform(df[task])

    for _, row in df.iterrows():
        uid = str(row['unique_id'])
        label = row[task]
        if row[group] in train_players:
            train_dict[uid] = label
        else:
            test_dict[uid] = label

    train_set = WaveformDataset(data_dir, train_dict, augment_fn=augment_fn)
    test_set = WaveformDataset(data_dir, test_dict)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = CNN1DClassifier(num_classes).to(device)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

    train_losses = []
    val_losses = []
    auc_scores = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            if num_classes > 2 and augment:
                x, y_a, y_b, lam = mixup_data(x, y)
                logits = model(x)
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                logits = model(x)
                loss = criterion(logits, y)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        train_losses.append(total_loss / len(train_loader))
        
        # 驗證階段
        model.eval()
        val_loss = 0
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for x_val, y_val in test_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                logits = model(x_val)
                loss = criterion(logits, y_val)
                val_loss += loss.item()
                probs = torch.softmax(logits, dim=1).cpu().numpy()
                all_preds.extend(probs)
                all_labels.extend(y_val.cpu().numpy())

        val_losses.append(val_loss / len(test_loader))
        # 轉換one-hot label
        y_true = np.array(all_labels)
        y_true_bin = label_binarize(y_true, classes=list(range(all_preds[0].shape[0])))
        try:
            if num_classes != 2:
                auc_score = roc_auc_score(y_true_bin, all_preds, average='micro', multi_class='ovr')
            else:
                auc_score = roc_auc_score(all_labels, [p[1] for p in all_preds], average='micro')  # 取出正類機率
        except ValueError as e:
            print(f"[AUC Error] {e}")
            auc_score = 0.0  # 若只有一類出現會錯誤
        auc_scores.append(auc_score)

        pred_classes = np.argmax(all_preds, axis=1)  # 從機率找出最大機率類別
        acc = accuracy_score(all_labels, pred_classes)

        print(f"Epoch {epoch+1}, Train Loss: {total_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(test_loader):.4f}, Val Accuracy: {acc:.4f}, AUC: {auc_score:.4f}")
        
        # pred_distribution = Counter(pred_classes)
        # print(f"Predicted class distribution: {dict(pred_distribution)}")



        if epoch == 0:
            best_val_loss = val_losses[-1]
            trigger = 0
        elif val_losses[-1] < best_val_loss:
            best_val_loss = val_losses[-1]
            trigger = 0
        else:
            trigger += 1
            if trigger >= 5:
                print("⛔ Early stopping triggered due to no improvement in validation loss.")
                break
        

    torch.save(model.state_dict(), f"weight/cnn1d_{task}.pt")
    print("✅ 訓練完成並儲存模型")

    # 畫出損失圖表
    epochs_ran = len(train_losses)
    plt.figure()
    plt.plot(range(1, epochs_ran + 1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs_ran + 1), val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'Training and Validation Loss - {task}')
    plt.legend()
    plt.savefig(f"result/loss_curve_{task}.png")
    print(f"📉 損失曲線已儲存為 loss_curve_{task}.png")

# ===== 使用範例 =====
if __name__ == '__main__':
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']
    # train_rnn("./train_data/", "train_info.csv", task="gender", group="player_id", num_classes=2,use_cuda=True, augment=False)
    # train_rnn("./train_data/", "train_info.csv", task="hold racket handed", group="mode", num_classes=2,use_cuda=True, augment=False)
    train_rnn("./train_data/", "train_info.csv", task="play years", group="player_id", num_classes=3,use_cuda=True, augment=True)
    train_rnn("./train_data/", "train_info.csv", task="level", group="player_id", num_classes=4,use_cuda=True, augment=True)
