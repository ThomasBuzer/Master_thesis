

from distutils.util import convert_path
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
import csv
import time
import matplotlib.pyplot as plt

class CNN(nn.Module):
    def __init__(self, target_number):
        super(CNN, self).__init__()

        self.target_number = target_number
        targets = [[1.0, 2.0, 4.0, 8.0, 16.0, 32.0], [2.0, 4.0, 8.0, 16.0, 32.0, 64.0], [1.0, 3.0, 5.0], [1.0, 2.0, 3.0], [0]]
        self.number_targets = len(targets[target_number])

        conv_layers=[]
        for i in range(4):
            conv_layers += [nn.Conv1d(min(2**i, 512), min(2**(i+1), 512), kernel_size=10, stride=8, padding=0),
            nn.BatchNorm1d(min(2**(i+1), 512)),
            nn.Dropout(0.5/(i+1)),
            nn.ReLU(inplace=True)]

        # for i in range(16):
        #     conv_layers += [nn.Conv1d(min(2**i, 512), min(2**(i+1), 512), kernel_size=3, stride=2, padding=0),
        #     nn.BatchNorm1d(min(2**(i+1), 512)),
        #     nn.Dropout(0.5/(i+1)),
        #     nn.ReLU(inplace=True)]


        #print(conv_layers)

        self.network = nn.Sequential(
            #*conv_layers,
            nn.Conv1d(1, 8, kernel_size=500, stride=50, padding=0),
            nn.BatchNorm1d(8),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),

            nn.Conv1d(8, 16, kernel_size=100, stride=50, padding=0),
            nn.BatchNorm1d(16),
            nn.Dropout(0.3),
            nn.ReLU(inplace=True),
            

            nn.Flatten(),
            nn.Linear(1568, 50),
            nn.Tanh(),
        
            # nn.Linear(480, 480),
            # nn.ReLU(inplace=True),
        
            nn.Linear(50, 1)

            
            )
    def forward(self, x):
        x = self.network(x)
        return x

class CustomImageDataset(torch.utils.data.Dataset):
    def __init__(self, x, y, target_number, transform=None, target_transform=None, stop=0):
        # self.img_labels = pd.read_csv(annotations_file)
        #self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform
        
        self.target_number = target_number
        self.len_img = 250000
        
        self.stop = stop
        if(not stop):
            self.stop = len(np.load(y))

        self.x = np.load(x)[:self.stop]
        #self.x = torch.tensor(np.load(x), dtype=torch.float)
        #self.x = torch.reshape(self.x, (len(self.x), 1, self.len_img))

        self.y = torch.tensor(np.load(y)[:,target_number], dtype=torch.float)[:self.stop]
        self.labels = torch.tensor(np.load(y), dtype=torch.float)[:self.stop]
        #self.y = torch.reshape(self.y, (len(self.y), len(self.y[0])))
        
        self.size = len(self.x)

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        #print(idx)
        image = torch.tensor(self.x[idx], dtype=torch.float)
        #print(image)
        #print(image.size())
        image = torch.reshape(image, (1, self.len_img))
        #image = self.x[idx]

        #print(image.size())
        label = self.y[idx]
        #print(label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

    def get_full_labels(self):
        return self.labels


def train(model, device, train_loader, optimizer, epoch, noise_amplitude=0):
    '''
    train the model
    '''
    model.train()
    counter = 0
    correct = 0
    print("Epoch "+str(epoch))
    for batch_idx, (data, target) in enumerate(train_loader):
        print("Batch N° " +str(counter))
        #print(data.size())
        data += torch.tensor((np.random.rand(len(data), 1, len(data[0]))-0.5)*2*noise_amplitude, dtype=torch.float)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = torch.flatten(model(data))
        # print(output)
        # print(target)
        loss = nn.MSELoss()(output, target)
        with open('./target_'+str(model.target_number)+ '/logs/train_loss.txt', 'a') as f:
                f.write(str(float(loss))+"\n")
        loss.backward()
        optimizer.step()
        counter += 1
        correct += quick_acc(output, target)
    acc = 100. * correct / len(train_loader.dataset) 
    print('\nTrain set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(train_loader.dataset), acc))
    with open('./target_'+str(model.target_number)+ '/logs/train_accuracy.txt', 'a') as f:
        f.write(str(correct)+"\n")

def quick_acc(output, target):
    size = len(output)
    counter = 0
    for i in range(size):
        if(abs(output[i] - target[i]) < 1+ (target[i]/10)):
            counter += 1
    # if(counter > size/2):
    #     return size
    # return 0
    return counter



def test(model, device, test_loader, full_labels = []):
    '''
    test the model
    '''

    target_array = [[1.0, 2.0, 4.0, 8.0, 16.0, 32.0], [2.0, 4.0, 8.0, 16.0, 32.0, 64.0], [1.0, 3.0, 5.0], [1.0, 2.0, 3.0], [0]]

    model.eval()
    test_loss = 0
    correct = 0
    counter = 0
    batch_size = 5
    table = np.zeros((6, 3, 3))
    hist = np.zeros((6, 200))
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            print("Test number " + str(counter))
            data, target = data.to(device), target.to(device)
            output = torch.flatten(model(data))
            #print(output)
            #print(output, target)
            # for i in range(len(target)):
            #     table[target[i]][pred[i][0]] += 1
            # loss = F.nll_loss(output, target)
            # with open('./target_'+str(model.target_number)+ '/logs/test_loss.txt', 'a') as f:
            #     f.write(str(float(loss))+"\n")

            batch_correct = quick_acc(output, target)
            correct += batch_correct
            counter += 1
            
            #for i in range(len(output)):
                #print(target[i], output[i])
                #hist[int(np.where(np.array(target_array[1]) == float(target[i]))[0][0])][int(output[i])] += 1

            # if(full_labels != []):
            #     #print(batch_correct, full_labels[batch_size*batch_idx])
            #     nin = int(np.where(np.array(target_array[0]) == float(full_labels[batch_size*batch_idx, 0]))[0][0])
            #     ksize = int(np.where(np.array(target_array[2]) == float(full_labels[batch_size*batch_idx, 2]))[0][0])
            #     stride = int(np.where(np.array(target_array[3]) == float(full_labels[batch_size*batch_idx, 3]))[0][0])
            #     table[nin][ksize][stride] += batch_correct

            #print(np.array(table))
            #table = [[0]*6 for i in range(6)]

            #if(counter%30 == 0):
            #    break
    with open('./target_'+str(model.target_number)+ '/logs/test_accuracy.txt', 'a') as f:
        f.write(str(correct)+"\n")
    acc = 100. * correct / len(test_loader.dataset)
    #np.save("target_1/logs/hists/hist_"+str(int(time.time()*100)), hist)
    #print(hist)
    print()
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_loader.dataset), acc))
    k=0
    for h in hist:
        plt.plot(h[:70], label = "Class " +str(k))
        k+=1
    plt.legend()
    plt.show()
    
    #np.save("reg_validation.npy", table)

    return


''' image transformation for training '''
train_transform = torchvision.transforms.Compose([
                           torchvision.transforms.RandomAffine(5,translate=(0.1,0.1)),
                           torchvision.transforms.ToTensor()
                           ])

''' image transformation for test '''
test_transform = torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor()
                           ])



''' image transformation for image generation '''
gen_transform = torchvision.transforms.Compose([
                           torchvision.transforms.ToTensor()
                           ])


