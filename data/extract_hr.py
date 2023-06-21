# import packages

# used heartpy 
import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import resample
import shutil

def viewData(file):
    print("Loading data from", file)
    npz = np.load(file, 'allow_pickle', True)
    x = npz['x']
    ann = npz['y']
    fs = npz['fs']
    ch_label = npz['ch_label']
    # start_datetime = npz['start_datetime']
    # file_duration = npz['file_duration']
    # epoch_duration = npz['epoch_duration']
    # n_all_epochs = npz['n_all_epochs']
    # n_epochs1 = npz['n_epochs'][0]; n_epochs2 = npz['n_epochs'][1]

    # print(file)
    # print(np.shape(x1), len(y1), fs1, ch_label1, n_epochs1)
    # print(np.shape(x2), len(y2), fs2, ch_label2, n_epochs2)
    # print(file_duration, epoch_duration, n_all_epochs, '\n')

    data = dict(npz)

    assert ch_label[0] == "ECG"
    assert fs[0] == fs[1] == fs[2] == fs[3] == fs[4]

    ecg = x[0]
    fs = fs[0]
    eeg_channels = x[1:]
    eeg_labels = ch_label[1:]
    
    return ecg, eeg_channels, ann, eeg_labels, fs, data


def heartRate(ecg, eeg, ann, fs):
    print("Entering heart rate extraction function")
    og_ecg = ecg
    original_shape = np.shape(ecg)
    ecg = np.concatenate(ecg)

    # for now
    # ecg = ecg[0:256*10]
    # eeg = eeg[0:256*10]
    # ann = ann[0:256*10]
    # ecg = hp.flip_signal(ecg)

    filtered = hp.filter_signal(ecg, cutoff = 0.05, sample_rate = fs, order = 3, filtertype='notch')

    #resample the data. Usually 2, 4, or 6 times is enough depending on original sampling rate
    factor = 10
    resampled_data = resample(filtered, len(filtered) * factor)

    print("Running ecg peak analysis")
    wd, m = hp.process(hp.scale_data(resampled_data), fs * factor)

    # wd = hp.analysis.calc_rr(wd['peaklist'], wd['sample_rate'], wd)

    faulty_idx = np.where(wd['binary_peaklist']==0)[0]
    # faulty_idx = np.ndarray.tolist(faulty_idx)
    peaklist = np.array(wd['peaklist'])
    peaklist[faulty_idx]=0

    heart_rate = np.zeros((len(peaklist),2))
    heart_rate_ecg = np.zeros(len(ecg)*factor)
    
    print("Computing heart rate from peak analysis")
    for i in range(1, len(peaklist)):
        if peaklist[i] == 0: continue
        if peaklist[i-1] != 0:
            diff = peaklist[i]-peaklist[i-1]
            heart_rate_ecg[peaklist[i]] = wd['sample_rate']/diff*60
        else: continue

    print("Shape conversion")
    heart_rate_ecg = np.reshape(heart_rate_ecg, (len(ann), int(wd['sample_rate'])))

    print("Removing epochs without heart_rate")
    heart_rate = np.zeros(len(heart_rate_ecg))
    for i in range(len(heart_rate_ecg)):
        hr_idx = np.where(heart_rate_ecg[i,:] != 0)
        if np.any(hr_idx) == True:
            heart_rate[i] = np.average(heart_rate_ecg[i, hr_idx])
  
    hr_idx = np.where(heart_rate != 0)

    ecg = np.reshape(ecg, original_shape)
    ecg = ecg[hr_idx]
    eeg = eeg[:,hr_idx]
    ann = ann[hr_idx]
    heart_rate = heart_rate[hr_idx]

    heart_rate = np.reshape(heart_rate, (len(heart_rate),1))
    ecg = np.reshape(ecg, (len(heart_rate), int(fs)))
    eeg = np.reshape(eeg, (4, len(heart_rate), int(fs)))

    # print(np.shape(heart_rate), np.shape(ecg), np.shape(eeg))
    heart_rate = np.ndarray.tolist(heart_rate)
    ecg = np.ndarray.tolist(ecg)
    eeg = np.ndarray.tolist(eeg)

    data = [heart_rate, ecg]
    for i in range(len(eeg)):
        data.append(eeg[i])

    return data, ann

input_dir = 'D:\hmc_prepared'
output_dir = 'D:\hr_extracted'

# Output dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
# else:
#     shutil.rmtree(output_dir)
#     os.makedirs(output_dir)

files = os.listdir(input_dir)

for f in range(len(files)):
    ecg, eeg_channels, ann, eeg_labels, fs, data = viewData(os.path.join(input_dir, files[f]))

    try:
        x, y = heartRate(ecg, eeg_channels, ann, fs)
    except hp.exceptions.BadSignalWarning as err:
        print("File thrown out due to error:", err)
    else:
        ch_labels = ["HR", "ECG"]
        for i in range(len(eeg_labels)):
            ch_labels.append(eeg_labels[i])

        save_dict = {
            "x": x, 
            "y": y,
            "fs": fs,
            "ch_label": ch_labels
        }
        print("Saving data")

        np.savez(os.path.join(output_dir, files[f]), **data)

        print("Data is now saved in new directory, containing 6 channels: hr, ecg and 4x eeg", "\n")
