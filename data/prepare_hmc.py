r"""
Adapted from https://github.com/akaraspt/tinysleepnet/blob/main/prepare_sleepedf.py    
"""

import argparse
import glob
import math
import ntpath
import os
import shutil
import pyedflib
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.io import savemat

# from extract_hr import heartRate
from sleepstage import stage_dict
from logger import get_logger
import logging


# Have to manually define based on the dataset
ann2label = {
    "Sleep stage W": 0,
    "Sleep stage N1": 1,
    "Sleep stage N2": 2,
    "Sleep stage N3": 3, "Sleep stage N4": 3, # Follow AASM Manual
    "Sleep stage R": 4,
    "Sleep stage ?": 6,
    "Movement time": 5
}

label2ann = {
    0: "Sleep stage W",
    1: "Sleep stage N1",
    2: "Sleep stage N2",
    3: "Sleep stage N3/4",
    4: "Sleep stage R"
}

# Deault params
DATA_DIR = "D:\HMC"
OUTPUT_DIR = "D:\hmc_prepared"
SELECT_CHANNEL = ["ECG", "EEG C4-M1", "EEG F4-M1", "EEG O2-M1", "EEG C3-M2"]
LOG_FILE = "info_ch_extract.log"


def main(
        data_dir: str = DATA_DIR,
        output_dir: str = OUTPUT_DIR,
        select_ch: str = SELECT_CHANNEL,
        log_file: str = LOG_FILE):
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    log_file = Path(log_file)

    # Output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    # else:
        # shutil.rmtree(output_dir)
        # os.makedirs(output_dir)

    log_file = os.path.join(output_dir, log_file)

    # Create logger
    # logger = get_logger(log_file, level="info")
    logging.basicConfig(format = '%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    

    # Read raw and annotation from EDF files
    psg_fnames = []; ann_fnames = []
    for i in os.listdir(data_dir):
        if len(i) == 9: psg_fnames.append(os.path.join(data_dir, i))
        if "sleepscoring.edf" in i: ann_fnames.append(os.path.join(data_dir, i))
    # psg_fnames = glob.glob(os.path.join(data_dir, i in data_dir if len(i)==9 ))
    # ann_fnames = glob.glob(os.path.join(data_dir, "*sleepscoring.df"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    w = 0; n1 = 0; n2 = 0; n34 = 0; r = 0

    for i in range(len(psg_fnames)):
        data_x = []; data_y = []
        sampling_rates = []
        n_epochs_all = []

        logger.info("Loading ...")
        logger.info("Signal file: {}".format(psg_fnames[i]))
        logger.info("Annotation file: {}".format(ann_fnames[i]))

        psg_f = pyedflib.EdfReader(psg_fnames[i])
        ann_f = pyedflib.EdfReader(ann_fnames[i])

        assert psg_f.getStartdatetime() == ann_f.getStartdatetime()
        start_datetime = psg_f.getStartdatetime()
        logger.info("Start datetime: {}".format(str(start_datetime)))

        file_duration = psg_f.getFileDuration()
        logger.info("File duration: {} sec".format(file_duration))
        epoch_duration = psg_f.datarecord_duration
        if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            epoch_duration = epoch_duration / 2
            logger.info("Epoch duration: {} sec (changed from 60 sec)".format(epoch_duration))
        else:
            logger.info("Epoch duration: {} sec".format(epoch_duration))

        # Generate labels from onset and duration annotation
        labels = []
        total_duration = 0
        ann_onsets, ann_durations, ann_stages = ann_f.readAnnotations()

        # Remove faulty annotations
        short_idx = np.where(ann_durations == 30)
        ann_onsets = ann_onsets[short_idx]
        ann_durations = ann_durations[short_idx]
        ann_stages = ann_stages[short_idx]

        for a in range(len(ann_stages)):
            onset_sec = int(ann_onsets[a])
            duration_sec = int(ann_durations[a])
            ann_str = "".join(ann_stages[a])

            # Sanity check
            assert onset_sec == total_duration

            # Get label value
            label = ann2label[ann_str]

            # Compute # of epoch for this stage
            if duration_sec % epoch_duration != 0:
                logger.info(f"Something wrong: {duration_sec} {epoch_duration}")
                raise Exception(f"Something wrong: {duration_sec} {epoch_duration}")
            duration_epoch = int(duration_sec / epoch_duration)

            # Generate sleep stage labels
            label_epoch = np.ones(duration_epoch, dtype=np.dtype(int)) * label
            labels.append(label_epoch)

            total_duration += duration_sec

            logger.info("Include onset:{}, duration:{}, label:{} ({})".format(
                onset_sec, duration_sec, label, ann_str
            ))
        labels = np.hstack(labels)

        # # count sleep stages
        # w = w + np.count_nonzero(labels==0)
        # n1 = n1 + np.count_nonzero(labels==1)
        # n2 = n2 + np.count_nonzero(labels==2)
        # n34 = n34 + np.count_nonzero(labels==3)
        # r = r + np.count_nonzero(labels==4)    

        for ch in range(len(select_ch)):
            # Extract signal from the selected channel
            ch_names = psg_f.getSignalLabels()
            ch_samples = psg_f.getNSamples()
            select_ch_idx = -1
            for s in range(psg_f.signals_in_file):
                if ch_names[s] == select_ch[ch]:
                    select_ch_idx = s
            if (select_ch_idx == -1):
                raise Exception("Channel not found.")
            
            sampling_rate = psg_f.getSampleFrequency(select_ch_idx)
            n_epoch_samples = int(epoch_duration * sampling_rate)
            signals = psg_f.readSignal(select_ch_idx).reshape(-1, n_epoch_samples)

            logger.info("Select channel: {}".format(select_ch))
            logger.info("Select channel samples: {}".format(ch_samples[select_ch_idx]))
            logger.info("Sample rate: {}".format(sampling_rate))

            # Sanity check
            n_epochs = psg_f.datarecords_in_file
            if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
                n_epochs = n_epochs * 2
            assert len(signals) == n_epochs, f"signal: {signals.shape} != {n_epochs}"

            # Remove annotations that are longer than the recorded signals
            labels = labels[:len(signals)]

            # Get epochs and their corresponding labels
            x = signals.astype(np.float32)
            y = labels.astype(np.int32)

            # Select only sleep periods
            w_edge_mins = 30
            nw_idx = np.where(y != stage_dict["W"])[0]
            start_idx = nw_idx[0] - (w_edge_mins * 2)
            end_idx = nw_idx[-1] + (w_edge_mins * 2)
            if start_idx < 0: start_idx = 0
            if end_idx >= len(y): end_idx = len(y) - 1
            select_idx = np.arange(start_idx, end_idx+1)

            logger.info("Data before selection: {}, {}".format(x.shape, y.shape))
            x = x[select_idx]
            y = y[select_idx]
            logger.info("Data before selection: {}, {}".format(x.shape, y.shape))

            data_x.append(x); data_y.append(y)
            sampling_rates.append(sampling_rate)
            n_epochs_all.append(len(x))

        # count sleep stages
        w = w + np.count_nonzero(y==0)
        n1 = n1 + np.count_nonzero(y==1)
        n2 = n2 + np.count_nonzero(y==2)
        n34 = n34 + np.count_nonzero(y==3)
        r = r + np.count_nonzero(y==4)


        # Save
        filename = ntpath.basename(psg_fnames[i]).replace(".edf", ".npz")
        save_dict = {
            "x": data_x, 
            "y": y,
            "fs": sampling_rates,
            "ch_label": select_ch,
            "start_datetime": start_datetime,
            "file_duration": file_duration,
            "epoch_duration": epoch_duration,
            "n_all_epochs": n_epochs,
            "n_epochs": n_epochs_all
        }

        np.savez(os.path.join(args.output_dir, filename), **save_dict)
        # print(os.path.join(args.output_dir, filename))
        logger.info("\n=======================================\n")

    print("W:", w, '\n', "N1:", n1, '\n', "N2:", n2, '\n', "N3/4:", n34, '\n', "R:", r)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch", type=str, default=SELECT_CHANNEL,
                        help="Name of the channel 2 in the datset."),
    parser.add_argument("--log_file", type=str, default=LOG_FILE,
                        help="Log file.")
    args = parser.parse_args()
    
    main(
        args.data_dir, 
        args.output_dir,
        args.select_ch,
        args.log_file)
