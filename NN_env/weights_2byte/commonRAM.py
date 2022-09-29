

from distutils.util import convert_path
import random
import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from tqdm import tqdm
import csv
import matplotlib.pyplot as plt
import matplotlib as mpl

NUMBER_CLASS = 256

class CNN(nn.Module):
    def __init__(self, target_number):
        super(CNN, self).__init__()

        self.target_number = target_number
        targets = [range(NUMBER_CLASS)]
        self.number_targets = len(targets[target_number])

        conv_layers=[]
        for i in range(3):
            conv_layers += [nn.Conv1d(min(2**i, 512), min(2**(i+1), 512), kernel_size=5, stride=2, padding=0),
            nn.BatchNorm1d(min(2**(i+1), 512)),
            nn.Dropout(0.5/(i+1)),
            nn.ReLU(inplace=True)]
        
        #print(conv_layers)

        self.network = nn.Sequential(
            #*conv_layers,
            nn.Conv1d(1, 4, kernel_size=10, stride=2, padding=0),
            nn.BatchNorm1d(4),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),

            
            nn.Conv1d(4, 8, kernel_size=700, stride=10, padding=0),
            nn.BatchNorm1d(8),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(4, stride=2),
            # nn.ReLU(inplace=True),
            nn.Flatten(),

            nn.Linear(492, 512),
            nn.Tanh(),
            nn.Dropout(0.2),
            
            # nn.Linear(512, 256),
            # nn.Tanh(),
            # nn.Linear(40, 16),
            # nn.ReLU(inplace=True),
            
            nn.Linear(512, self.number_targets)
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
        self.len_img = 7998
        targets = [range(256)]
        self.target_array = np.array(targets[target_number])
        
        self.stop = stop
        if(not stop):
            self.stop = len(np.load(y))

        self.x = np.load(x)[:self.stop]
        #self.x = torch.tensor(np.load(x), dtype=torch.float)
        #self.x = torch.reshape(self.x, (len(self.x), 1, self.len_img))

        self.y = torch.tensor(np.load(y)[:,target_number], dtype=torch.int64)[:self.stop]
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
        #print(np.where(self.target_array == float(label)))
        label = int(np.where(self.target_array == float(label))[0][0])
        #print(label)
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def train(model, device, train_loader, optimizer, epoch, noise_amplitude=0):
    '''
    train the model
    '''
    model.train()
    counter = 0
    correct = 0
    print("Epoch "+str(epoch))
    for batch_idx, (data, target) in enumerate(train_loader):
        #print("Batch N° " +str(counter))
        #print(data.size())
        data += torch.tensor((np.random.rand(len(data), 1, len(data[0]))-0.5)*2*noise_amplitude, dtype=torch.float)#data augmentation 1
        #data = torch.roll(data, random.randint(-8, 8))#data augmentation 2
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        x = model(data)

        # output = F.log_softmax(input=x,dim=0)
        # loss = F.nll_loss(output, target)
        
        #
        # output = x.argmax(dim=1)
        # mseloss = nn.MSELoss()
        # loss= mseloss(output, target.flatten())
        # 
        output = x
        cross_loss = nn.CrossEntropyLoss()
        loss = cross_loss(output, target)

        with open('./target_'+str(model.target_number)+ '/logs/train_loss.txt', 'a') as f:
                f.write(str(float(loss))+"\n")
        loss.backward()
        optimizer.step()
        counter += 1
        #if(counter%10 == 0):
        #    break
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
    acc = 100. * correct / len(train_loader.dataset) 
    print('\nTrain set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(train_loader.dataset), acc))
    with open('./target_'+str(model.target_number)+ '/logs/train_accuracy.txt', 'a') as f:
        f.write(str(correct)+"\n")



def test(model, device, test_loader, plot = False):
    '''
    test the model
    '''
    model.eval()
    test_loss = 0
    correct = 0
    counter = 0
    
    error = 0
    table = np.array([[0]*NUMBER_CLASS for i in range(NUMBER_CLASS)])
    with torch.no_grad():
        for data, target in test_loader:
            #print("Test number " + str(counter))
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            for i in range(len(target)):
                table[target[i], pred[i][0]] += 1

            
            error += torch.sum(torch.absolute(torch.flatten(pred) - torch.flatten(target)))
            
            correct += pred.eq(target.view_as(pred)).sum().item()
            counter +=1
    with open('./target_'+str(model.target_number)+ '/logs/test_accuracy.txt', 'a') as f:
        f.write(str(correct)+"\n")
    with open('./target_'+str(model.target_number)+ '/logs/test_loss.txt', 'a') as f:
               f.write(str(int(error))+"\n")
    acc = 100. * correct / len(test_loader.dataset)
    print('\nTest set: Accuracy: {}/{} ({:.2f}%)\n'.format(correct, len(test_loader.dataset), acc))
    
    if plot:
        # plt.figure(figsize=(7, 8))
        
        # plt.xlabel("Guessed class")
        # plt.ylabel("Real class")

        # cmap = (mpl.colors.ListedColormap(['goldenrod', 'darkorange', 'darkred'])
        # .with_extremes(over='0.25', under='0.75'))
        # cmap.set_extremes(under='white', over='darkgreen')
        # bounds = [2, 5, 20, 45]
        # norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
        ismax = table == np.max(table, axis=1, keepdims=True)
        print(np.sum(np.diag(ismax)))
        # plt.pcolormesh(table, cmap=cmap, norm=norm)

        # plt.colorbar(
        #                 mpl.cm.ScalarMappable(cmap=cmap, norm=norm),
        #                 #cax=ax,
        #                 boundaries=[0] + bounds,  # Adding values for extensions.
        #                 extend='both',
        #                 ticks=bounds,
        #                 spacing='proportional',
        #             )
        # plt.show()

    return

def shift_test(data, model):
    max_shift = 50
    shift_step = 1
    table = torch.zeros((5,3))
    for shift in range(-max_shift, max_shift+1, shift_step):
        output = model(torch.cat((data[:,:,shift:], data[:,:,:shift]), 2))
        #print("output", output.size(), output)
        pred = output.argmax(dim=1)
        #print("pred", pred.size(), pred)
        for i,p in enumerate(pred):
            table[i, p] += 1
        
    print("table", table.size(), table)
    return table.argmax(dim=1, keepdim=True)

def noise_test(data, model):
    max_noise = 10
    noise_step = 1
    table = torch.zeros((5,3))
    for noise_amp in range(-max_noise, max_noise, noise_step):
        noise = torch.rand((5,1, 250000))*noise_amp/10
        output = model( data+noise)
        #print("output", output.size(), output)
        pred = output.argmax(dim=1)
        #print("pred", pred.size(), pred)
        for i,p in enumerate(pred):
            table[i, p] += 1
        
    print("table", table.size(), table)
    return table.argmax(dim=1, keepdim=True)



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


