import os
import pandas as pd
import torch
import torch.nn.functional as F
import numpy as np
from train_rnn import CNN1DClassifier, WaveformDataset
from sklearn.preprocessing import LabelEncoder

# ====== 任務設定 ======
target_info = {
    'gender': {'task_type': 'binary', 'columns': ['gender']},
    'hold racket handed': {'task_type': 'binary', 'columns': ['hold racket handed']},
    'play years': {
        'task_type': 'multiary',
        'columns': ['play years_0', 'play years_1', 'play years_2']
    },
    'level': {
        'task_type': 'multiary',
        'columns': ['level_2', 'level_3', 'level_4', 'level_5']
    }
}

# ====== 推論單一筆資料用 ======
def read_waveform_txt(file_path, num_segments=27, target_length=100):
    with open(file_path) as f:
        lines = f.readlines()[1:]
        data = [list(map(int, line.strip().split())) for line in lines if line.strip() and len(line.strip().split()) == 6]
    data = np.array(data)
    total_len = len(data)
    segment_idx = np.linspace(0, total_len, num_segments + 1, dtype=int)
    segments = []
    for i in range(num_segments):
        seg = data[segment_idx[i]:segment_idx[i + 1]]
        if len(seg) >= target_length:
            seg = seg[:target_length]
        else:
            pad = np.zeros((target_length - len(seg), 6))
            seg = np.vstack([seg, pad])
        segments.append(seg)
    result = np.stack(segments)
    return torch.tensor(result, dtype=torch.float32)  # shape: (27, 100, 6)

# ====== 主推論流程 ======
def test_rnn(test_info_path, test_data_dir, model_dir, submission_template, output_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    test_info = pd.read_csv(test_info_path)
    submission = pd.read_csv(submission_template)
    predictions = {col: [] for col in submission.columns if col != 'unique_id'}

    for task, info in target_info.items():
        print(f"\n📌 Running task: {task}")
        model_path = os.path.join(model_dir, f"cnn1d_{task}.pt")
        num_classes = len(info['columns'])
        if(num_classes == 1):
            num_classes = 2

        model = CNN1DClassifier(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        for uid in test_info['unique_id']:
            path = os.path.join(test_data_dir, f"{uid}.txt")
            if not os.path.exists(path):
                probs = [0.0] * num_classes
            else:
                tensor = read_waveform_txt(path).unsqueeze(0).to(device)  # shape: (1, 27, 100, 6)
                with torch.no_grad():
                    logits = model(tensor)
                    prob = F.softmax(logits, dim=1).squeeze(0).cpu().numpy()
                    prob = np.round(prob.astype(np.float64), 10)  # 防止 scientific notation
            # 寫入預測機率
            if num_classes == 2 and len(info['columns']) == 1:
                # binary 任務只寫出正類機率 (index 0)
                predictions[info['columns'][0]].append(float(prob[0]) if isinstance(prob, (list, np.ndarray)) else 0.0)
            else:
                for col, p in zip(info['columns'], prob):
                    predictions[col].append(p)


    # 組合結果
    for col in submission.columns:
        if col == 'unique_id':
            submission[col] = test_info['unique_id']
        else:
            submission[col] = predictions[col]

    submission.to_csv(output_path, index=False, float_format='%.10f')
    print(f"\n✅ Submission saved to {output_path}")

# ===== 使用示例 =====
if __name__ == '__main__':
    test_rnn(
        test_info_path='test_info.csv',
        test_data_dir='test_data/',
        model_dir='weight',
        submission_template='sample_submission.csv',
        output_path='submission.csv'
    )
