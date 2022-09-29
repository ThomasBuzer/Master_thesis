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

from commonRAM import *


DIVIDER = '-----------------------------------------'

test_csv="test.csv"

def test_only(build_dir, batchsize, model_path, target_number):

    x_test = build_dir + "/x_val_small.npy"
    y_test = build_dir + "/y_val.npy"

    float_model = build_dir + "/validation"    

    targets = [[1.0, 2.0, 4.0, 8.0, 16.0, 32.0], [2.0, 4.0, 8.0, 16.0, 32.0, 64.0], [1.0, 3.0, 5.0], [1.0, 2.0, 3.0], [0]]

    
    # use GPU if available   
    print('No CUDA devices available..selecting CPU')
    device = torch.device('cpu')

    model = CNN(target_number).to(device)
    print("Loading model into memory")
    model.load_state_dict(torch.load(os.path.join(float_model,model_path)))


    #image datasets
    print("Loading images into memory")
    test_dataset = CustomImageDataset(x_test, y_test, target_number)

    #data loaders
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batchsize, 
                                              shuffle=False)


    # test model 
    test(model, device, test_loader)


    return


def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir',   type=str,  default='.',       help='Path to build folder. Default is build')
    ap.add_argument('-m', '--model',   type=str,  default='f_model.pth',           help='Path to the model to test')
    ap.add_argument('-b', '--batchsize',   type=int,  default=100,           help='Batch size ofc')
    ap.add_argument('-t','--target',   type=int,default=3,         help='Position of the target in the csv file. Default is 3')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('PyTorch version : ',torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--build_dir    : ',args.build_dir)
    print ('--batchsize    : ', args.batchsize)
    print ('--model    : ',args.model)
    print ('--target       : ',args.target)
    print(DIVIDER)

    test_only(args.build_dir, args.batchsize, args.model, args.target)

    return



if __name__ == '__main__':
    run_main()
