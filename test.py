from pathlib import Path
import numpy as np
import pandas as pd
import math
import csv
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

import joblib
import sys
from preprocess import *

def data_generate():
    datapath = './test_data'
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

    # read info from files
    info = pd.read_csv('test_info.csv')
    unique_ids = info['unique_id'].tolist()

    # read from data
    print('reading features')
    feature_dir = './tabular_data_train'
    X_list = []
    feature_dim = 34

    for uid in unique_ids:
        path = Path(feature_dir) / f'{uid}.csv'
        if path.exists():
            df = pd.read_csv(path)
        else:
            df = pd.DataFrame()

        if len(df) == 0:
            df = pd.DataFrame(np.zeros((27, feature_dim)),
                              columns=[f'f{i}' for i in range(feature_dim)])
        elif len(df) < 27:
            pad_rows = 27 - len(df)
            pad = pd.DataFrame(
                np.repeat(df.tail(1).values, pad_rows, axis=0),
                columns=df.columns
            )
            df = pd.concat([df, pad], ignore_index=True)
        elif len(df) > 27:
            df = df.iloc[:27]

        X_list.append(df.values)

    X_pool = np.vstack(X_list)

    # scale
    scaler = joblib.load('../scaler.m')
    feature_names = scaler.feature_names_in_ # fix : "UserWarning: X does not have valid feature names"
    df_pool = pd.DataFrame(X_pool, columns=feature_names)
    X_pool_scaled = scaler.transform(df_pool)

    group_size = 27
    current_idx = 0

    def model_binary(X_test):
        # get saved model
        nonlocal current_idx
        clf = joblib.load('../model-{}.m'.format(current_idx))
        predicted = clf.predict_proba(X_test)
        eps=1e-7
        predicted = predicted[:, 1]
        logit = np.log(np.clip(predicted, eps, 1 - eps) / np.clip(1 - predicted, eps, 1 - eps))
        num_groups = len(logit) // group_size
        y_pred = []
        for i in range(num_groups):
            gp = logit[i*group_size : (i+1)*group_size]
            cur = np.mean(gp)
            y_pred.append(1/(1 + np.exp(-cur)))

        current_idx += 1
        return y_pred

    # 定義多類別分類評分函數 (例如 play years、level)
    def model_multiary(X_test):
        nonlocal current_idx
        clf = joblib.load('../model-{}.m'.format(current_idx))
        predicted = clf.predict_proba(X_test)
        num_groups = len(predicted) // group_size
        y_pred = predicted.reshape(num_groups, group_size, -1).mean(axis=1)

        current_idx += 1
        return y_pred

    # generate result
    print('predict from trained model')
    sample_submission = pd.read_csv('./sample_submission.csv')
    results = pd.DataFrame()
    results['unique_id'] = sample_submission['unique_id']

    gender_pred = model_binary(X_pool_scaled)
    hold_pred = model_binary(X_pool_scaled)
    age_pred = model_multiary(X_pool_scaled)
    level_pred = model_multiary(X_pool_scaled)

    # unique_id,gender,hold racket handed,play years_0,play years_1,play years_2,level_2,level_3,level_4,level_5
    age_arr = np.vstack(age_pred)
    level_arr = np.vstack(level_pred)
    results['gender'] = gender_pred
    results['hold racket handed'] = hold_pred
    results['play years_0'] = age_arr[:, 0]
    results['play years_1'] = age_arr[:, 1]
    results['play years_2'] = age_arr[:, 2]
    results['level_2'] = level_arr[:, 0]
    results['level_3'] = level_arr[:, 1]
    results['level_4'] = level_arr[:, 2]
    results['level_5'] = level_arr[:, 3]

    results.to_csv('submission.csv', index=False)
    print('finish')

if __name__ == '__main__':
    main()
