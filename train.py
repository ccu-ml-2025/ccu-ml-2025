from pathlib import Path
import pandas as pd
from joblib import dump #儲存權重與模型
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from model import model_binary_xgb, model_multiary_xgb
from features import data_generate

def main():
    # 若尚未產生特徵，請先執行 data_generate() 生成特徵 CSV 檔案
    data_generate('train')

    # 讀取訓練資訊，根據 player_id 將資料分成 80% 訓練、20% 測試
    info = pd.read_csv('train_info.csv')
    unique_players = info['player_id'].unique()
    train_players, test_players = train_test_split(unique_players, test_size=0.2, random_state=42)

    # 讀取特徵 CSV 檔（位於 "./tabular_data_train"）
    datapath = './tabular_data_train'
    datalist = list(Path(datapath).glob('**/*.csv'))
    target_mask = ['gender', 'hold racket handed', 'play years', 'level']

    # 根據 test_players 分組資料
    x_train = pd.DataFrame()
    y_train = pd.DataFrame(columns=target_mask)
    x_test = pd.DataFrame()
    y_test = pd.DataFrame(columns=target_mask)

    for file in datalist:
        unique_id = int(Path(file).stem)
        row = info[info['unique_id'] == unique_id]
        if row.empty:
            continue
        player_id = row['player_id'].iloc[0]
        data = pd.read_csv(file)
        target = row[target_mask]
        target_repeated = pd.concat([target] * len(data))
        if player_id in train_players:
            x_train = pd.concat([x_train, data], ignore_index=True)
            y_train = pd.concat([y_train, target_repeated], ignore_index=True)
        elif player_id in test_players:
            x_test = pd.concat([x_test, data], ignore_index=True)
            y_test = pd.concat([y_test, target_repeated], ignore_index=True)

    # 標準化特徵
    scaler = MinMaxScaler()
    le = LabelEncoder()
    X_train_scaled = scaler.fit_transform(x_train)
    X_test_scaled = scaler.transform(x_test)

    group_size = 27 #27次揮拍



    # 評分：針對各目標進行模型訓練與評分
    y_train_le_gender = le.fit_transform(y_train['gender'])
    y_test_le_gender = le.transform(y_test['gender'])
    model_binary_xgb(X_train_scaled, y_train_le_gender, X_test_scaled, y_test_le_gender, group_size, 'gender')

    y_train_le_hold = le.fit_transform(y_train['hold racket handed'])
    y_test_le_hold = le.transform(y_test['hold racket handed'])
    model_binary_xgb(X_train_scaled, y_train_le_hold, X_test_scaled, y_test_le_hold, group_size, 'hold')

    y_train_le_years = le.fit_transform(y_train['play years'])
    y_test_le_years = le.transform(y_test['play years'])
    model_multiary_xgb(X_train_scaled, y_train_le_years, X_test_scaled, y_test_le_years, group_size, 'years')

    y_train_le_level = le.fit_transform(y_train['level'])
    y_test_le_level = le.transform(y_test['level'])
    model_multiary_xgb(X_train_scaled, y_train_le_level, X_test_scaled, y_test_le_level, group_size, 'levels')

    #AUC SCORE: 0.792(gender) + 0.998(hold) + 0.660(years) + 0.822(levels)

    dump(scaler, './models/scaler.joblib')
    dump(le, './models/le.joblib')

if __name__ == '__main__':
    main()
