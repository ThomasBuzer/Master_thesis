import argparse
from wsgiref import headers
import trsfile
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
import time
import random
import os

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--file', type=str, default='./traces/big/file.trs', help='Path to trs file. Default is ./traces/trace.trs')
args = ap.parse_args()

window_size = int(1000)
step_size = int(window_size/10)

x_test = np.load("./x_test.npy")
y_test = np.load("./y_test.npy")

x_train = np.load("./x_train.npy")
y_train = np.load("./y_train.npy")

trace_dir = "./traces/big"

n_traces = 350
trace_length = 250000
n_train_per_file = 280
n_test_per_file = 70


while 1:
    files = os.listdir(trace_dir)
    files = [os.path.join(trace_dir,file) for file in files]
    if files != []:
        max_file = max(files, key= lambda x: os.stat(x).st_size) 

    #print(len(files), max_file)

    #Wait for 2 files to be in so that at least 1 of them is full
    if(len(files) < 2):
        time.sleep(5)
        continue

    

    with trsfile.open(max_file, 'r') as traces:
        # Show all headers
        # for header, value in traces.get_headers().items():
        #     print(header, '=', value)
            
        # print()
        
        print("processing : ", max_file)
        for i in tqdm(range(len(traces))):
            trace = traces[i]
            new_trace= []
            n_train = 0
            n_test = 0

            #ABS and WIN_RES
            for j in (range(int((len(trace)-window_size)/step_size))):
                result = np.average(np.absolute(trace[step_size*j:step_size*j+window_size]))
                new_trace += [result]
            #0 padding or crop
            new_trace = new_trace[:trace_length] + [0]*(trace_length- len(new_trace))
            #print([new_trace[:]])

            #ADD to train / test file
            if((random.random() < 0.8 or n_test == n_test_per_file) and n_train < n_train_per_file):
                x_train = np.append(x_train, [new_trace[:]], axis=0)
                y_train = np.append(y_train, [max_file.split('_')[1:6]], axis=0)
                n_train += 1
            else:
                x_test = np.append(x_test, [new_trace[:]], axis=0)
                y_test = np.append(y_test, [max_file.split('_')[1:6]], axis=0) 
                n_test += 1


        #save the files
        np.save("./x_train.npy", x_train)
        np.save("./y_train.npy", y_train)

        np.save("./x_test.npy", x_test)
        np.save("./y_test.npy", y_test)

        #remove trs
        os.remove(max_file)







