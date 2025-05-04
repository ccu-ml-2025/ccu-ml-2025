from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score, make_scorer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, PredefinedSplit
from sklearn.model_selection import StratifiedGroupKFold

import joblib
import sys
from preprocess import *

def data_generate():
    datapath = './train_data'
    tar_dir = 'tabular_data_train'
    pathlist_txt = Path(datapath).glob('**/*.txt')


    for file in pathlist_txt:
        f = open(file)

        All_data = []

        count = 0
        for line in f.readlines():
            if line == '\n' or count == 0:
                count += 1
                continue
            num = line.split(' ')
            if len(num) > 5:
                tmp_list = []
                for i in range(6):
                    tmp_list.append(int(num[i]))
                All_data.append(tmp_list)

        f.close()

        swing_index = np.linspace(0, len(All_data), 28, dtype=int)
        headerList = ['ax_mean', 'ay_mean', 'az_mean', 'gx_mean', 'gy_mean', 'gz_mean', 'ax_var', 'ay_var', 'az_var', 'gx_var', 'gy_var', 'gz_var', 'ax_rms', 'ay_rms', 'az_rms', 'gx_rms', 'gy_rms', 'gz_rms', 'a_max', 'a_mean', 'a_min', 'g_max', 'g_mean', 'g_min', 'a_fft', 'g_fft', 'a_psd', 'g_psd', 'a_kurt', 'g_kurt', 'a_skewn', 'g_skewn', 'a_entropy', 'g_entropy']

        with open('./{dir}/{fname}.csv'.format(dir = tar_dir, fname = Path(file).stem), 'w', newline = '') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headerList)
            a_fft, g_fft = FFT_data(All_data, swing_index)
            a_fft_imag = [0] * len(a_fft)
            g_fft_imag = [0] * len(g_fft)
            n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
            n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
            for i in range(len(swing_index)):
                if i==0:
                    continue
                feature(All_data[swing_index[i-1]: swing_index[i]], i - 1, len(swing_index) - 1, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag, writer)


def main():
    # 若尚未產生特徵，請先執行 data_generate() 生成特徵 CSV 檔案
    data_generate()

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

    group_size = 27
    final_scores = [0 for _ in range(4)]
    current_idx = 0

    def binary_group_auc(y_test, predicted):
        eps=1e-7
        predicted = predicted[:, 1]
        logit = np.log(np.clip(predicted, eps, 1 - eps) / np.clip(1 - predicted, eps, 1 - eps))
        num_groups = len(logit) // group_size
        y_pred = []
        for i in range(num_groups):
            gp = logit[i*group_size : (i+1)*group_size]
            cur = np.mean(gp)
            y_pred.append(1/(1 + np.exp(-cur)))

        y_test_agg = y_test[::group_size][:num_groups]
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro')
        return auc_score

    def binary_group_scorer(estimator, X, y_true):
        y_pred = estimator.predict_proba(X)
        return binary_group_auc(y_true, y_pred)

    def model_binary(X_train, y_train, X_test, y_test):
        # merge training set & valid set
        X_pool = np.vstack((X_train, X_test))
        Y_pool = np.concatenate((y_train, y_test))
        # tell PredefinedSplit where to train(-1) and valid(0)
        n_train, n_test = len(X_train), len(X_test)
        test_fold = np.concatenate([np.full(n_train, -1, dtype=int), np.zeros(n_test, dtype=int)])
        cv = PredefinedSplit(test_fold)

        params = {
            'n_neighbors' : list(range(1, 55, 2)),
            'metric': ['euclidean', 'manhattan']
        }
        grid = GridSearchCV(
            estimator = KNeighborsClassifier(),
            param_grid = params,
            cv = cv, # Use pre-defined split (8:2)
            scoring = binary_group_scorer,
            n_jobs = -1,
            verbose = 2,
            error_score=np.nan,
            refit = False # no leak
        )

        grid.fit(X_pool, Y_pool)
        best_param = grid.best_params_
        clf = KNeighborsClassifier(**best_param)
        clf.fit(X_train, y_train)
        predicted = clf.predict_proba(X_test)
        nonlocal current_idx
        final_scores[current_idx] = binary_group_auc(y_test, predicted)

        # save model into outfile
        joblib.dump(clf, '../model-{}.m'.format(current_idx))
        current_idx += 1

    def multiary_group_auc(y_test, predicted):
        num_groups = len(predicted) // group_size
        y_pred = predicted.reshape(num_groups, group_size, -1).mean(axis=1)
        y_test_agg = y_test[::group_size][:num_groups]
        auc_score = roc_auc_score(y_test_agg, y_pred, average='micro', multi_class='ovr')
        return auc_score

    def multiary_group_scorer(estimator, X, y_true):
        y_pred = estimator.predict_proba(X)
        return multiary_group_auc(y_true, y_pred)

    def model_multiary(X_train, y_train, X_test, y_test):
        # merge training set & valid set
        X_pool = np.vstack((X_train, X_test))
        Y_pool = np.concatenate((y_train, y_test))
        # tell PredefinedSplit where to train(-1) and valid(0)
        n_train, n_test = len(X_train), len(X_test)
        test_fold = np.concatenate([np.full(n_train, -1, dtype=int), np.zeros(n_test, dtype=int)])
        cv = PredefinedSplit(test_fold)

        params = {
            'n_neighbors' : list(range(1, 55, 2)),
            'metric': ['euclidean', 'manhattan']
        }
        grid = GridSearchCV(
            estimator = KNeighborsClassifier(),
            param_grid = params,
            cv = cv, # Use pre-defined split (8:2)
            scoring = multiary_group_scorer,
            n_jobs = -1,
            verbose = 2,
            error_score=np.nan,
            refit = False # no leak
        )

        grid.fit(X_pool, Y_pool)
        best_param = grid.best_params_
        clf = KNeighborsClassifier(**best_param)
        clf.fit(X_train, y_train)
        predicted = clf.predict_proba(X_test)
        nonlocal current_idx
        final_scores[current_idx] = multiary_group_auc(y_test, predicted)

        # save model into outfile
        joblib.dump(clf, '../model-{}.m'.format(current_idx))
        current_idx += 1

    # 評分：針對各目標進行模型訓練與評分
    y_train_le_gender = le.fit_transform(y_train['gender'])
    y_test_le_gender = le.transform(y_test['gender'])
    model_binary(X_train_scaled, y_train_le_gender, X_test_scaled, y_test_le_gender)

    y_train_le_hold = le.fit_transform(y_train['hold racket handed'])
    y_test_le_hold = le.transform(y_test['hold racket handed'])
    model_binary(X_train_scaled, y_train_le_hold, X_test_scaled, y_test_le_hold)

    y_train_le_years = le.fit_transform(y_train['play years'])
    y_test_le_years = le.transform(y_test['play years'])
    model_multiary(X_train_scaled, y_train_le_years, X_test_scaled, y_test_le_years)

    y_train_le_level = le.fit_transform(y_train['level'])
    y_test_le_level = le.transform(y_test['level'])
    model_multiary(X_train_scaled, y_train_le_level, X_test_scaled, y_test_le_level)

    #AUC SCORE: 0.792(gender) + 0.998(hold) + 0.660(years) + 0.822(levels)
    for i in range(4):
        print('Binary AUC:' if i<2 else 'Multiary AUC:', end=' ')
        print(final_scores[i])

    # save the scaler
    joblib.dump(scaler, '../scaler.m')

if __name__ == '__main__':
    main()
