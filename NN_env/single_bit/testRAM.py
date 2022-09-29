

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

def test_only(build_dir, batchsize, model_path, target_number):

    x_test = build_dir + "/x_test.npy"
    y_test = build_dir + "/y_test.npy"

    float_model = build_dir + "/validation"    

    targets = [range(256)]
    
    # use GPU if available   
    print('No CUDA devices available..selecting CPU')
    device = torch.device('cpu')

    model = CNN(target_number).to(device)
    print("Loading model into memory")

    model.load_state_dict(torch.load(os.path.join(float_model,model_path)))
    # model = torch.load(os.path.join(float_model,model_path))
    # print(model)

    #image datasets
    print("Loading images into memory")
    test_dataset = CustomImageDataset(x_test, y_test, target_number)

    #data loaders
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batchsize, 
                                              shuffle=False)


    # test model 
    test(model, device, test_loader, plot=True, bit = target_number)


    return


def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir',   type=str,  default='.',       help='Path to build folder. Default is build')
    ap.add_argument('-m', '--model',   type=str,  default='f_model.pth',           help='Path to the model to test')
    ap.add_argument('-b', '--batchsize',   type=int,  default=100,           help='Batch size ofc')
    ap.add_argument('-t','--target',   type=int,default=0,         help='Position of the target in the csv file. Default is 3')
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
