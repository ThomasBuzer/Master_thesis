'''
 Copyright 2020 Xilinx Inc.
 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at
     http://www.apache.org/licenses/LICENSE-2.0
 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
'''


'''
Simple PyTorch MNIST example - training & testing
'''

'''
Author: Mark Harvey, Xilinx inc
'''

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import sys
import os
import shutil
import random

from commonRAM import *

import trsfile
import numpy as np
import matplotlib.pyplot as plt

DIVIDER = '-----------------------------------------'


def infer_only(build_dir, batchsize):

    float_model = build_dir +"/float_model"   
    
    targets = [[1.0], [8.0, 10.0, 12.0, 14.0, 16.0], [3.0, 5.0], [1.0, 2.0, 3.0], [0]]

    models_name = [
        "f_model_target_1.pth",
        "f_model_target_2.pth",
        "f_model_target_3.pth"
    ]

    #file_trs = build_dir+"/traces/layer_1_16_3_1_0_350_1657012601.trs"
    #file_trs = build_dir+"/traces/layer_1_16_3_2_0_350_1657012671.trs"
    #file_trs = build_dir+"/traces/layer_1_16_3_3_0_350_1657012741.trs"
    #file_trs = build_dir+"/traces/layer_1_16_5_1_0_350_1657012810.trs"
    #file_trs = build_dir+"/traces/layer_1_16_5_2_0_350_1657012880.trs"
    #file_trs = build_dir+"/traces/layer_1_16_5_3_0_350_1657012950.trs"
    #file_trs = build_dir+"/traces/layer_1_12_5_1_0_1000_1656676450.trs"
    #file_trs = build_dir+"/traces/layer_1_12_3_2_0_1000_1656676328.trs"
    #file_trs = build_dir+"/traces/layer_1_8_3_1_0_350_1657012028.trs"
    file_trs = build_dir+"/traces/layer_1_8_5_1_0_350_1657012239.trs"

    
    

    # use GPU if available   
    if (torch.cuda.device_count() > 0):
        print('You have',torch.cuda.device_count(),'CUDA devices available')
        for i in range(torch.cuda.device_count()):
            print(' Device',str(i),': ',torch.cuda.get_device_name(i))
        print('Selecting device 0..')
        device = torch.device('cuda:0')
    else:
        print('No CUDA devices available..selecting CPU')
    device = torch.device('cpu')

    model_one = CNN(len(targets[1])).to(device)
    model_two = CNN(len(targets[2])).to(device)
    model_three = CNN(len(targets[3])).to(device)

    model_one.load_state_dict(torch.load(os.path.join(float_model, models_name[0])))
    model_two.load_state_dict(torch.load(os.path.join(float_model, models_name[1])))
    model_three.load_state_dict(torch.load(os.path.join(float_model, models_name[2])))

    model_one.eval()
    model_two.eval()
    model_three.eval()

    window_size = 30000
    step_size = window_size // 100
    
    counter = 0
    with trsfile.open(file_trs, 'r') as traces:
        # Show all headers
        for header, value in traces.get_headers().items():
            print(header, '=', value)
        print()
        table = np.zeros((3, 5))
        for trace in traces:
            #trace = traces[i]
            print(trace)
            print(len(trace))
            if(len(trace) < window_size):
                trace += [0]*(window_size-len(trace)) 
            n_steps = (len(trace) - window_size + step_size) // step_size 
            print(n_steps)
            print("Trace : ", counter)
            input = np.zeros([n_steps, 1, window_size])
            output = torch.ones([n_steps, 3, 5])*-100
            for j in range(n_steps):
                input[j] = trace[j*step_size:j*step_size+window_size]
                #plt.plot(input[j, 0])
                #plt.show()
            print(input)
            with torch.no_grad():
                for i in range(n_steps//batchsize+bool(n_steps//batchsize)):
                    print("Step : ", i)
                    input_tensor = torch.tensor(input[i*batchsize:min((i+1)*batchsize, n_steps)], dtype=torch.float)
                    output[i*batchsize:min((i+1)*batchsize, n_steps), 0, :] = model_one(input_tensor)
                    output[i*batchsize:min((i+1)*batchsize, n_steps), 1, :2] = model_two(input_tensor)
                    output[i*batchsize:min((i+1)*batchsize, n_steps), 2, :3] = model_three(input_tensor)

            #print(output[:, 1, :])

            #convert output to probability
            output = torch.nn.functional.softmax(output, dim=2)
            
            # extract the argmax
            pred = output.argmax(dim=2, keepdim=True)
            print(pred)

            #table[0,pred[:, 0]] += 1
            #table[1,pred[:, 1]] += 1
            #table[2,pred[:, 2]] += 1
            #print(table)
            plt.plot(pred[:, 0])
            #print(output[:, 0, :])
            plt.show()
            plt.plot(pred[:, 1])
            #print(output[:, 1, :])
            plt.show()
            plt.plot(pred[:, 2])
            #print(output[:, 2, :])
            plt.show()
            counter += 1
    return


def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir',   type=str,  default='.',       help='Path to build folder. Default is build')
    #ap.add_argument('-m', '--model',   type=str,  default='f_model.pth',           help='Path to the model to test')
    ap.add_argument('-b', '--batchsize',   type=int,  default=10,           help='Batch size ofc Default is 100')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('PyTorch version : ',torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--build_dir    : ',args.build_dir)
    print ('--batchsize    : ', args.batchsize)
    #print ('--model    : ',args.model)
    print(DIVIDER)

    infer_only(args.build_dir, args.batchsize)

    return



if __name__ == '__main__':
    run_main()
