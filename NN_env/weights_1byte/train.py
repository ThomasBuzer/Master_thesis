

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import argparse
import sys
import os
import shutil
from tqdm import tqdm

from commonRAM import *


DIVIDER = '-----------------------------------------'


def train_test(build_dir, batchsize, learnrate, epochs, target_number, model_path):

    target_dir = '/target_'+str(target_number)
    float_model = build_dir+ target_dir  + '/float_model'
 
    targets = [range(NUMBER_CLASS)]


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

    model = CNN(target_number).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learnrate)

    if(model_path != ''):
        model.load_state_dict(torch.load(os.path.join(float_model,model_path)))


    x_test = build_dir + "/x_test.npy"
    y_test = build_dir + "/y_test.npy"

    x_train = build_dir + "/x_train.npy"
    y_train = build_dir + "/y_train.npy"
    

    #image datasets
    train_dataset = CustomImageDataset(x_train, y_train, target_number)
    test_dataset = CustomImageDataset(x_test, y_test, target_number)

    

    #data loaders
    train_loader = torch.utils.data.DataLoader(train_dataset,
                                               batch_size=batchsize, 
                                               shuffle=True)
    test_loader = torch.utils.data.DataLoader(test_dataset,
                                              batch_size=batchsize, 
                                              shuffle=False)

  
    
    
    # training with test after each epoch
    for epoch in tqdm(range(1, epochs + 1)):
        
        train(model, device, train_loader, optimizer, epoch, noise_amplitude=0)
        test(model, device, test_loader)
        save_path = float_model + '/f_model_target_'+str(target_number)+'_temp'+'.pth'
        torch.save(model.state_dict(), save_path)

    # save the trained model
    save_path = float_model + '/f_model_target_'+str(target_number)+'.pth'
    torch.save(model.state_dict(), save_path) 
    print('Trained model written to',save_path)

    os.remove(float_model + '/f_model_target_'+str(target_number)+'_temp'+'.pth')

    return


def run_main():

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument('-d', '--build_dir',   type=str,  default='.',       help='Path to build folder. Default is build')
    ap.add_argument('-b', '--batchsize',   type=int,  default=100,           help='Training batchsize. Must be an integer. Default is 100')
    ap.add_argument('-e', '--epochs',      type=int,  default=3,             help='Number of training epochs. Must be an integer. Default is 3')
    ap.add_argument('-lr','--learnrate',   type=float,default=0.001,         help='Optimizer learning rate. Must be floating-point value. Default is 0.001')
    ap.add_argument('-t','--target',   type=int,default=0,         help='Position of the target in the csv file. Default is 3')
    ap.add_argument('-m','--model',   type=str,default='',         help='Name of the model file to use. Default is None')
    args = ap.parse_args()

    print('\n'+DIVIDER)
    print('PyTorch version : ',torch.__version__)
    print(sys.version)
    print(DIVIDER)
    print(' Command line options:')
    print ('--build_dir    : ',args.build_dir)
    print ('--batchsize    : ',args.batchsize)
    print ('--learnrate    : ',args.learnrate)
    print ('--epochs       : ',args.epochs)
    print ('--target       : ',args.target)
    print ('--model       : ',args.model)
    print(DIVIDER)

    train_test(args.build_dir, args.batchsize, args.learnrate, args.epochs, args.target, args.model)

    return



if __name__ == '__main__':
    run_main()
