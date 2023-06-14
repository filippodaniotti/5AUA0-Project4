# import packages

# used heartpy 
import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import resample
import shutil

def viewData(file):
    print("Loading data")
    npz = np.load(file, 'allow_pickle', True)
    x1 = npz['x1']; x2 = npz['x2']
    y1 = npz['y1']; y2 = npz['y2']
    fs1 = npz['fs'][0]; fs2 = npz['fs'][1]
    ch_label1 = npz['ch_label'][0]; ch_label2 = npz['ch_label'][1]
    # start_datetime = npz['start_datetime']
    file_duration = npz['file_duration']
    epoch_duration = npz['epoch_duration']
    n_all_epochs = npz['n_all_epochs']
    n_epochs1 = npz['n_epochs'][0]; n_epochs2 = npz['n_epochs'][1]

    # print(file)
    # print(np.shape(x1), len(y1), fs1, ch_label1, n_epochs1)
    # print(np.shape(x2), len(y2), fs2, ch_label2, n_epochs2)
    # print(file_duration, epoch_duration, n_all_epochs, '\n')
    data = dict(npz)
    return (x1,y1,fs1, ch_label1), (x2, y2, fs2, ch_label2), data


def heartRate(ecg, eeg, ann, fs):
    print("Entering heart rate extraction function")
    original_shape = np.shape(ecg)
    ecg = np.concatenate(ecg)

    # for now
    # ecg = ecg[0:256*10]
    # ann = ann[0:256*10]

    # ecg = hp.flip_signal(ecg)

    filtered = hp.filter_signal(ecg, cutoff = 0.05, sample_rate = fs, order = 3, filtertype='notch')

    #resample the data. Usually 2, 4, or 6 times is enough depending on original sampling rate
    factor = 10
    resampled_data = resample(filtered, len(filtered) * factor)

    print("Running ecg peak analysis")
    #And run the analysis again. Don't forget to up the sample rate as well!
    wd, m = hp.process(hp.scale_data(resampled_data), fs * factor)

    # #visualise in plot of custom size
    # hp.plotter(wd, m)
    # plt.show()

    #display computed measures
    # for measure in m.keys():
    #     print('%s: %f' %(measure, m[measure]))
    
    # print(wd.keys())
    wd = hp.analysis.calc_rr(wd['peaklist'], wd['sample_rate'], wd)

    faulty_idx = np.where(wd['binary_peaklist']==0)
    peaklist = wd['peaklist']
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
            heart_rate[i] = np.average(heart_rate_ecg[i,hr_idx])
  
    hr_idx = np.where(heart_rate != 0)

    ecg = ecg[hr_idx]
    eeg = eeg[hr_idx]
    ann = ann[hr_idx]
    heart_rate = heart_rate[hr_idx]

    return ecg, eeg, heart_rate, ann


input_dir = '..\dataset\hmc'
output_dir = '..\dataset\hr'

# Output dir
if not os.path.exists(output_dir):
    os.makedirs(output_dir)
else:
    shutil.rmtree(output_dir)
    os.makedirs(output_dir)

files = os.listdir(input_dir)

for f in range(len(files)):
    (x1,y1,fs1, ch_label1), (x2, y2, fs2, ch_label2), data = viewData(os.path.join(input_dir, files[f]))
    assert ch_label1 == "ECG"
    assert np.all(y1 == y2)

    ecg, eeg, hr, ann = heartRate(x1, x2, y1, fs1)

    print("Saving data")
    data['x1'] = ecg
    data['y1'] = ann
    data['x2'] = eeg
    data['y2'] = ann
    data['x3'] = hr
    data['y3'] = ann
    np.savez(os.path.join(output_dir, files[f]), **data)

    print("Data is now saved in new directory, containing 3 channels: ecg, eeg and hr")
