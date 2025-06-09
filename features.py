from fft_utils import FFT, FFT_data
import numpy as np
from scipy.stats import skew, kurtosis, entropy
import csv
from pathlib import Path
from tqdm import tqdm
from joblib import Parallel, delayed
import sys

headerList = [
    'ax_mean','ax_std','ax_rms','ax_max','ax_mean_dup','ax_min','ax_kurtosis','ax_skew','ax_energy',
    'ay_mean','ay_std','ay_rms','ay_max','ay_mean_dup','ay_min','ay_kurtosis','ay_skew','ay_energy',
    'az_mean','az_std','az_rms','az_max','az_mean_dup','az_min','az_kurtosis','az_skew','az_energy',
    'gx_mean','gx_std','gx_rms','gx_max','gx_mean_dup','gx_min','gx_kurtosis','gx_skew','gx_energy',
    'gy_mean','gy_std','gy_rms','gy_max','gy_mean_dup','gy_min','gy_kurtosis','gy_skew','gy_energy',
    'gz_mean','gz_std','gz_rms','gz_max','gz_mean_dup','gz_min','gz_kurtosis','gz_skew','gz_energy',
    'acc_max','acc_mean','acc_min','gyro_max','gyro_mean','gyro_min',
    'a_top1','a_top2','a_top3','a_top4','a_top5','a_max_freq','a_energy','a_entropy',
    'g_top1','g_top2','g_top3','g_top4','g_top5','g_max_freq','g_energy','g_entropy',
    'hori_acc_mean', 'hori_acc_std', 'hori_gyro_mean', 'hori_gyro_std',
    'ang_xy_mean', 'ang_xy_std', 'ang_yz_mean', 'ang_yz_std',
    'acc_x_rate', 'acc_y_rate', 'acc_z_rate',
    'gyro_x_rate', 'gyro_y_rate', 'gyro_z_rate',
    'acc_x_peak', 'acc_y_peak', 'acc_z_peak'
]

def compute_time_features(signal):
    std_val = np.std(signal)
    if std_val < 1e-8:
        kurt = 0.0
        skw = 0.0
    else:
        kurt = kurtosis(signal)
        skw = skew(signal)

    return [
        np.mean(signal),
        np.std(signal),
        np.sqrt(np.mean(signal**2)),
        np.max(signal),
        np.mean(signal),
        np.min(signal),
        kurt,
        skw,
        np.sum(signal ** 2)
    ]

def spectral_features(mag):
    psd = mag ** 2
    psd_norm = psd / np.sum(psd)
    top_k_amp = np.sort(mag)[-5:][::-1]
    max_freq = np.argmax(mag)
    energy = np.sum(psd)
    spec_entropy = entropy(psd_norm)
    return list(top_k_amp) + [max_freq, energy, spec_entropy]

def extract_features(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag):
    input_array = np.array(input_data)
    Ax, Ay, Az = input_array[:, 0].astype(float), input_array[:, 1].astype(float), input_array[:, 2].astype(float)
    Gx, Gy, Gz = input_array[:, 3].astype(float), input_array[:, 4].astype(float), input_array[:, 5].astype(float)

    # 6*9 features
    ax_feat = compute_time_features(Ax)
    ay_feat = compute_time_features(Ay)
    az_feat = compute_time_features(Az)
    gx_feat = compute_time_features(Gx)
    gy_feat = compute_time_features(Gy)
    gz_feat = compute_time_features(Gz)

    total_acc = np.sqrt(Ax**2 + Ay**2 + Az**2)
    total_gyro = np.sqrt(Gx**2 + Gy**2 + Gz**2)
    # 2*3 features
    acc_feat = compute_time_features(total_acc)
    gyro_feat = compute_time_features(total_gyro)

    cut = int(n_fft / swinging_times)
    a_mag = np.sqrt(np.array(a_fft[cut * swinging_now:cut * (swinging_now + 1)])**2 +
                    np.array(a_fft_imag[cut * swinging_now:cut * (swinging_now + 1)])**2)
    g_mag = np.sqrt(np.array(g_fft[cut * swinging_now:cut * (swinging_now + 1)])**2 +
                    np.array(g_fft_imag[cut * swinging_now:cut * (swinging_now + 1)])**2)

    # 2*8 features
    a_freq_feat = spectral_features(a_mag)
    g_freq_feat = spectral_features(g_mag)

    output = (
        ax_feat + ay_feat + az_feat +
        gx_feat + gy_feat + gz_feat +
        acc_feat[3:6] + gyro_feat[3:6] +
        a_freq_feat + g_freq_feat
    )
    output = np.nan_to_num(output, nan=0.0, posinf=0.0, neginf=0.0).tolist()

    return output

def extract_features_enhanced(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag):
    basic_features = extract_features(input_data, swinging_now, swinging_times, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag)

    input_array = np.array(input_data)
    Ax, Ay, Az = input_array[:, 0].astype(float), input_array[:, 1].astype(float), input_array[:, 2].astype(float)
    Gx, Gy, Gz = input_array[:, 3].astype(float), input_array[:, 4].astype(float), input_array[:, 5].astype(float)

    hori_acc = np.sqrt(Ax**2 + Ay**2)
    hori_gyro = np.sqrt(Gx**2 + Gy**2)
    ang_xy = np.arctan2(Ay, Ax)
    ang_yz = np.arctan2(Az, Ay)
    acc_rate = [np.mean(np.abs(np.diff(Ax))), np.mean(np.abs(np.diff(Ay))), np.mean(np.abs(np.diff(Az)))]
    gyro_rate = [np.mean(np.abs(np.diff(Gx))), np.mean(np.abs(np.diff(Gy))), np.mean(np.abs(np.diff(Gz)))]
    sig_len = len(Ax)
    acc_peaks = [np.argmax(np.abs(Ax))/sig_len, np.argmax(np.abs(Ay))/sig_len, np.argmax(np.abs(Az))/sig_len]

    extra_features = [
        np.mean(hori_acc), np.std(hori_acc),
        np.mean(hori_gyro), np.std(hori_gyro),
        np.mean(ang_xy), np.std(ang_xy),
        np.mean(ang_yz), np.std(ang_yz),
        *acc_rate,
        *gyro_rate,
        *acc_peaks
    ]

    return basic_features + extra_features

def process_file(file, task):
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

    swing_index = np.linspace(0, len(All_data), 28, dtype = int)

    with open('./{dir}/{fname}.csv'.format(dir='tabular_data_{}'.format(task), fname=Path(file).stem), 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(headerList)
        try:
            a_fft, g_fft = FFT_data(All_data, swing_index)
            a_fft_imag = [0] * len(a_fft)
            g_fft_imag = [0] * len(g_fft)
            n_fft, a_fft, a_fft_imag = FFT(a_fft, a_fft_imag)
            n_fft, g_fft, g_fft_imag = FFT(g_fft, g_fft_imag)
            for i in range(len(swing_index)):
                if i==0:
                    continue
                features = extract_features_enhanced(All_data[swing_index[i-1]: swing_index[i]], i - 1, len(swing_index) - 1, n_fft, a_fft, g_fft, a_fft_imag, g_fft_imag)
                writer.writerow(features)
        except Exception:
            print(Path(file).stem)
            return False
    return True

def data_generate(task):
    datapath = './{}_data'.format(task)
    pathlist_txt = list(Path(datapath).glob('**/*.txt'))

    Path('tabular_data_{}'.format(task)).mkdir(exist_ok=True)

    results = Parallel(n_jobs=-1)(
        delayed(process_file)(file, str(task)) for file in tqdm(pathlist_txt, desc='Generating features({})'.format(task), file=sys.stdout)
    )

    print(f"Generation success: {sum(results)} / {len(pathlist_txt)}")
