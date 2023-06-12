# import packages

# used heartpy 
import heartpy as hp
import matplotlib.pyplot as plt
import numpy as np
import os
from scipy.signal import resample

def viewData(file):
    npz = np.load(file)
    x1 = npz['x1']; x2 = npz['x2']
    y1 = npz['y1']; y2 = npz['y2']
    fs1 = npz['fs'][0]; fs2 = npz['fs'][1]
    ch_label1 = npz['ch_label'][0]; ch_label2 = npz['ch_label'][1]
    # start_datetime = npz['start_datetime']
    file_duration = npz['file_duration']
    epoch_duration = npz['epoch_duration']
    n_all_epochs = npz['n_all_epochs']
    n_epochs1 = npz['n_epochs'][0]; n_epochs2 = npz['n_epochs'][1]

    print(file)
    print(np.shape(x1), len(y1), fs1, ch_label1, n_epochs1)
    print(np.shape(x2), len(y2), fs2, ch_label2, n_epochs2)
    print(file_duration, epoch_duration, n_all_epochs, '\n')

    return (x1,y1,fs1, ch_label1), (x2, y2, fs2, ch_label2)


def heartRate(ecg, ann, fs):
    original_shape = np.shape(ecg)
    ecg = np.concatenate(ecg)
    
    # for now
    ecg = ecg[0:100000]

    # ecg = hp.flip_signal(ecg)

    filtered = hp.filter_signal(ecg, cutoff = 0.05, sample_rate = fs, order = 3, filtertype='notch')

    #resample the data. Usually 2, 4, or 6 times is enough depending on original sampling rate
    factor = 10
    resampled_data = resample(filtered, len(filtered) * factor)

    #And run the analysis again. Don't forget to up the sample rate as well!
    wd, m = hp.process(hp.scale_data(resampled_data), fs * factor)

    #visualise in plot of custom size
    hp.plotter(wd, m)
    plt.show()

    #display computed measures
    for measure in m.keys():
        print('%s: %f' %(measure, m[measure]))

    print(wd['removed_beats'])
    # print(wd.keys())
    # print(wd['peaklist'])



data_dir = '..\dataset\hmc'
files = os.listdir(data_dir)

for f in range(1): #len(files)):
    (x1,y1,fs1, ch_label1), (x2, y2, fs2, ch_label2) = viewData(os.path.join(data_dir, files[f]))
    assert ch_label1 == "ECG"
    
    heartRate(x1, y1, fs1)


