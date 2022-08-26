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

trs_path = '../layers_trs/'

x_trainfile_name="./x_train"
y_trainfile_name="./y_train"
x_testfile_name="./x_test"
y_testfile_name="./y_test"

trs_files = [f for f in os.listdir(trs_path) if os.path.isfile(os.path.join(trs_path, f))]
n_files = len(trs_files)
n_traces = 150
n_train_per_file = 120
n_test_per_file = 30

x_train = np.zeros(shape=(int(0.8*n_traces*n_files), 250000), dtype=np.int8)
y_train = np.zeros(shape=(int(0.8*n_traces*n_files), 5))
x_test = np.zeros(shape=(int(0.2*n_traces*n_files), 250000), dtype=np.int8)
y_test = np.zeros(shape=(int(0.2*n_traces*n_files), 5))
n_train = 0
n_test = 0

for i in tqdm(range(len(trs_files))):
    f = trs_files[i]
    #print(f)
    counter = 0
    with trsfile.open(trs_path+ f, 'r') as traces:
        # Show all headers
        #for header, value in traces.get_headers().items():
        #    print(header, '=', value)
        #print()
        n_train = 0
        n_test = 0
        for trace in traces:
            if((random.random() < 0.8 or n_test == n_test_per_file) and n_train < n_train_per_file):
                x_train[i*n_train_per_file+n_train] = trace[:]
                y_train[i*n_train_per_file+n_train] = f.split('_')[1:6]
                n_train += 1
            else:
                x_test[i*n_test_per_file+n_test] = trace[:]
                y_test[i*n_test_per_file+n_test] = f.split('_')[1:6]
                n_test += 1
            counter += 1
        #print(f, " Done !")
print("Done !")

np.save(x_trainfile_name, x_train)
np.save(y_trainfile_name, y_train)

np.save(x_testfile_name, x_test)
np.save(y_testfile_name, y_test)

print("All Saved !")
