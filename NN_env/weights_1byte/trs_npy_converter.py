import argparse
from csv import writer
import time
import trsfile
import numpy as np
from multiprocessing import Pool, current_process
import matplotlib.pyplot as plt
from tqdm import tqdm
import os
import random

# ap = argparse.ArgumentParser()
# ap.add_argument('-f', '--file', type=str, default='./traces/trace.trs', help='Path to trs file. Default is ./traces/trace.trs')
# ap.add_argument('-n', '--name', type=str, default='peak', help='Name of the files. Default is :with')
# args = ap.parse_args()

#name=args.name

trs_path = '../../pico/traces/singleW/resampled_1byte'

x_trainfile_name="./x_train"
y_trainfile_name="./y_train"
x_testfile_name="./x_test"
y_testfile_name="./y_test"

trs_files = [f for f in os.listdir(trs_path) if os.path.isfile(os.path.join(trs_path, f))]
n_files = len(trs_files)
n_traces = 500
n_train_per_file = 80*5
n_test_per_file = 20*5
n_samples = 7998
threshold = 12

x_train = np.zeros((1, n_samples)) #np.zeros(shape=(int(0.8*n_traces*n_files), n_samples), dtype=np.float)
y_train = np.zeros((1, 1)) #np.zeros(shape=(int(0.8*n_traces*n_files), 1))
x_test = np.zeros((1, n_samples)) #np.zeros(shape=(int(0.2*n_traces*n_files), n_samples), dtype=np.float)
y_test = np.zeros((1, 1)) #np.zeros(shape=(int(0.2*n_traces*n_files), 1))
#print(y_train.shape)
n_train = 0
n_test = 0



for i in tqdm(range(len(trs_files))):
    f = trs_files[i]
    #print(f)
    with trsfile.open(os.path.join(trs_path, f), 'r') as traces:
        
        x_train_buf = np.zeros((1, n_samples))
        y_train_buf = np.zeros((1, 1))
        x_test_buf = np.zeros((1, n_samples))
        y_test_buf = np.zeros((1, 1))
        
        n_train = 0
        n_test = 0
        for trace in traces:
            mean = np.sum(np.array(trace[:]))/n_samples
            #print(mean)
            if( mean <= threshold):
                continue
            if((random.random() < 0.8 or n_test == n_test_per_file) and n_train < n_train_per_file):
                #print(x_train.shape, y_train.shape)
                x_train_buf = np.concatenate((x_train_buf, np.array([trace[:]])), axis=0)
                y_train_buf = np.concatenate((y_train_buf, np.array([[f.split('_')[1]]])), axis=0)
                n_train += 1
            else:
                #print(x_test.shape, y_test.shape)
                x_test_buf = np.concatenate((x_test_buf, np.array([trace[:]])), axis=0)
                y_test_buf = np.concatenate((y_test_buf, np.array([[f.split('_')[1]]])), axis=0)
                n_test += 1
        x_train = np.concatenate((x_train, x_train_buf[1:]), axis=0)
        y_train = np.concatenate((y_train, y_train_buf[1:]), axis=0)
        x_test = np.concatenate((x_test, x_test_buf[1:]), axis=0)
        y_test = np.concatenate((y_test, y_test_buf[1:]), axis=0)
        #print(f, " Done !")
print("Done !")

np.save(x_trainfile_name, x_train[1:])
np.save(y_trainfile_name, y_train[1:])

np.save(x_testfile_name, x_test[1:])
np.save(y_testfile_name, y_test[1:])

print("All Saved !")
