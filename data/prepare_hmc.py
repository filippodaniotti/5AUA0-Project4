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
OUTPUT_DIR = "..\dataset\hmc"
SELECT_CHANNEL_1 = "ECG"
SELECT_CHANNEL_2 = "EEG C4-M1"
LOG_FILE = "info_ch_extract.log"


def main(
        data_dir: str = DATA_DIR,
        output_dir: str = OUTPUT_DIR,
        select_ch1: str = SELECT_CHANNEL_1,
        select_ch2: str = SELECT_CHANNEL_2,
        log_file: str = LOG_FILE):
    
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    log_file = Path(log_file)

    # Output dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    log_file = os.path.join(output_dir, log_file)

    # Create logger
    # logger = get_logger(log_file, level="info")
    logging.basicConfig(format = '%(levelname)s: %(message)s')
    logger = logging.getLogger(__name__)
    

    # Read raw and annotation from EDF files
    psg_fnames = []; ann_fnames = []
    for i in os.listdir(data_dir):
        if len(i) == 9: psg_fnames.append(os.path.join(data_dir, i))
        if "sleepscoring" in i: ann_fnames.append(os.path.join(data_dir, i))
    # psg_fnames = glob.glob(os.path.join(data_dir, i in data_dir if len(i)==9 ))
    # ann_fnames = glob.glob(os.path.join(data_dir, "*sleepscoring.df"))
    psg_fnames.sort()
    ann_fnames.sort()
    psg_fnames = np.asarray(psg_fnames)
    ann_fnames = np.asarray(ann_fnames)

    for i in range(len(psg_fnames)):

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

        # Extract signal from the selected channel
        ch_names = psg_f.getSignalLabels()
        ch_samples = psg_f.getNSamples()
        select_ch_idx_1 = -1
        select_ch_idx_2 = -1
        for s in range(psg_f.signals_in_file):
            if ch_names[s] == select_ch1:
                select_ch_idx_1 = s
            if ch_names[s] == select_ch2:
                select_ch_idx_2 = s
        if (select_ch_idx_1 == -1) or (select_ch_idx_2 == -1):
            raise Exception("Channel not found.")
        
        sampling_rate_1 = psg_f.getSampleFrequency(select_ch_idx_1)
        n_epoch_samples_1 = int(epoch_duration * sampling_rate_1)
        signals_1 = psg_f.readSignal(select_ch_idx_1).reshape(-1, n_epoch_samples_1)
        
        sampling_rate_2 = psg_f.getSampleFrequency(select_ch_idx_2)
        n_epoch_samples_2 = int(epoch_duration * sampling_rate_2)
        signals_2 = psg_f.readSignal(select_ch_idx_2).reshape(-1, n_epoch_samples_2)

        logger.info("Select channel: {}".format(select_ch1))
        logger.info("Select channel samples: {}".format(ch_samples[select_ch_idx_1]))
        logger.info("Sample rate: {}".format(sampling_rate_1))

        logger.info("Select channel: {}".format(select_ch2))
        logger.info("Select channel samples: {}".format(ch_samples[select_ch_idx_2]))
        logger.info("Sample rate: {}".format(sampling_rate_2))

        # Sanity check
        n_epochs = psg_f.datarecords_in_file
        if psg_f.datarecord_duration == 60: # Fix problems of SC4362F0-PSG.edf, SC4362FC-Hypnogram.edf
            n_epochs = n_epochs * 2
        assert len(signals_1) == n_epochs, f"signal: {signals_1.shape} != {n_epochs}"
        assert len(signals_2) == n_epochs, f"signal: {signals_2.shape} != {n_epochs}"

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

        # Remove annotations that are longer than the recorded signals
        labels_1 = labels[:len(signals_1)]
        labels_2 = labels[:len(signals_2)]

        # Get epochs and their corresponding labels
        x1 = signals_1.astype(np.float32)
        x2 = signals_2.astype(np.float32)
        y1 = labels_1.astype(np.int32)
        y2 = labels_2.astype(np.int32)

        # Select only sleep periods
        w_edge_mins = 30
        nw_idx1 = np.where(y1 != stage_dict["W"])[0]
        nw_idx2 = np.where(y2 != stage_dict["W"])[0]
        start_idx1 = nw_idx1[0] - (w_edge_mins * 2)
        start_idx2 = nw_idx2[0] - (w_edge_mins * 2)
        end_idx1 = nw_idx1[-1] + (w_edge_mins * 2)
        end_idx2 = nw_idx2[-1] + (w_edge_mins * 2)
        if start_idx1 < 0: start_idx1 = 0
        if start_idx2 < 0: start_idx2 = 0
        if end_idx1 >= len(y1): end_idx1 = len(y1) - 1
        if end_idx2 >= len(y2): end_idx2 = len(y2) - 1
        select_idx1 = np.arange(start_idx1, end_idx1+1)
        select_idx2 = np.arange(start_idx2, end_idx2+1)

        logger.info("Data before selection: {}, {}, {}, {}".format(x1.shape, y1.shape, x2.shape, y2.shape))
        x1 = x1[select_idx1]
        x2 = x2[select_idx2]
        y1 = y1[select_idx1]
        y2 = y2[select_idx2]
        logger.info("Data before selection: {}, {}, {}, {}".format(x1.shape, y1.shape, x2.shape, y2.shape))
        
        # # Remove movement and unknown
        # move_idx = np.where(y1 == stage_dict["MOVE"])[0]
        # unk_idx = np.where(y == stage_dict["UNK"])[0]
        # if len(move_idx) > 0 or len(unk_idx) > 0:
        #     remove_idx = np.union1d(move_idx, unk_idx)
        #     logger.info("Remove irrelavant stages")
        #     logger.info("  Movement: ({}) {}".format(len(move_idx), move_idx))
        #     logger.info("  Unknown: ({}) {}".format(len(unk_idx), unk_idx))
        #     logger.info("  Remove: ({}) {}".format(len(remove_idx), remove_idx))
        #     logger.info("  Data before removal: {}, {}, {}, {}".format(x1.shape, y1.shape, x2.shape, y2.shape))
        #     select_idx = np.setdiff1d(np.arange(len(x1)), remove_idx)
        #     x1 = x1[select_idx]
        #     x2 = x2[select_idx]
        #     y = y[select_idx]
        #     logger.info("  Data after removal: {}, {}, {}".format(x1.shape, x2.shape, y.shape))
        
        # Save
        filename = ntpath.basename(psg_fnames[i]).replace(".edf", ".npz")
        save_dict = {
            "x1": x1, 
            "x2": x2,
            "y1": y1,
            "y2": y2, 
            "fs": [sampling_rate_1, sampling_rate_2],
            "ch_label": [select_ch1, select_ch2],
            "start_datetime": start_datetime,
            "file_duration": file_duration,
            "epoch_duration": epoch_duration,
            "n_all_epochs": n_epochs,
            "n_epochs": [len(x1), len(x2)],
        }

        np.savez(os.path.join(args.output_dir, filename), **save_dict)
        print(os.path.join(args.output_dir, filename))
        logger.info("\n=======================================\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default=DATA_DIR,
                        help="File path to the Sleep-EDF dataset.")
    parser.add_argument("--output_dir", type=str, default=OUTPUT_DIR,
                        help="Directory where to save outputs.")
    parser.add_argument("--select_ch1", type=str, default=SELECT_CHANNEL_1,
                        help="Name of the channel 1 in the dataset.")
    parser.add_argument("--select_ch2", type=str, default = SELECT_CHANNEL_2,
                        help="Name of the channel 2 in the datset."),
    parser.add_argument("--log_file", type=str, default=LOG_FILE,
                        help="Log file.")
    args = parser.parse_args()
    
    main(
        args.data_dir, 
        args.output_dir,
        args.select_ch1,
        args.select_ch2,
        args.log_file)
