import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import torchvision
normalize = False

#Defines the dataset class
class Dataset(object):
    def __init__(self, n):
        X = np.load('data/train_images.npy')
        Y = np.load('data/train_labels.npy')
        self.len = X.shape[0]
        self.x_data = torch.from_numpy(X).float()
        self.y_data = torch.from_numpy(Y).float()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if (normalize):
            curr = self.x_data[idx].numpy()
            mean = np.mean(self.x_data[idx].numpy())
            std = np.std(self.x_data[idx].numpy())
            return (curr - mean / std, self.y_data[idx])
        return self.x_data[idx], self.y_data[idx]

class DatasetTest(object):
    def __init__(self, n):
      X = np.load('data/test_images.npy')
      Y = np.load('data/test_labels.npy')
      self.len = X.shape[0]
      self.x_data = torch.from_numpy(X).float()
      self.y_data = torch.from_numpy(Y).float()

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        if (normalize):
            curr = self.x_data[idx].numpy()
            mean = np.mean(self.x_data[idx].numpy())
            std = np.std(self.x_data[idx].numpy())
            return (curr - mean / std, self.y_data[idx])
        return self.x_data[idx], self.y_data[idx]

#Defines the model for experiment one
class NetExperimentOne(nn.Module):
    def __init__(self):
        super(NetExperimentOne, self).__init__()
        self.fc1 = nn.Linear(32 * 32 * 3, 1024)
        self.fc2 = nn.Linear(1024, 10)

    def forward(self, x):
        x = x.view(x.size(0),3*32*32)
        out = F.sigmoid(self.fc1(x))
        out = F.sigmoid(self.fc2(out))
        return out

#Defines the model for experiment two
class NetExperimentTwo(nn.Module):
    def __init__(self):
        super(NetExperimentTwo, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5, 1, 0)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
        self.fc1 = nn.Linear(16 * 10 * 10, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
        #self.conv1 = nn.Conv2d(3, 6, 5, 1, 0)
       # self.pool = nn.MaxPool2d(2, 2)
        #self.conv2 = nn.Conv2d(6, 16, 5, 1, 0)
      #  self.fc1 = nn.Linear(16 * 10 * 10, 120)
       # self.fc2 = nn.Linear(120, 84)
      #  self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        out = F.relu(self.conv1(x))
        out = self.pool(out)
        out = F.relu(self.conv2(out))
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = F.sigmoid(self.fc3(out))
        return out


def runExperiment(name, lr, batch_size, momentum, shuffle, max_epochs, n):
    if (name is 'NetExperimentOne'):
        net = NetExperimentOne()
    elif (name is 'NetExperimentTwo'):
        net = NetExperimentTwo()
    criterion = nn.CrossEntropyLoss()
    dataset = Dataset(n)
    train_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle)
    optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    accuracy = np.zeros(max_epochs)
    for epoch in range(max_epochs):
        correct = 0
        size = 0
        totalLoss = 0
        iters = 0
        for i, data in enumerate(train_loader, 0):
            inputs, labels = data
            inputs, labels = Variable(inputs), Variable(labels).long()
            y_pred = net(inputs)
            loss = criterion(y_pred, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            y_pred_np = y_pred.data.numpy()
            pred_np = np.argmax(y_pred_np, axis=1)
            label_np = labels.data.numpy().reshape(len(labels), 1)
            for j in range(y_pred_np.shape[0]):
                size += 1
                if pred_np[j] == label_np[j,:]:
                    correct += 1
            totalLoss += loss.data
            iters += 1
        accuracy[epoch] = float(correct) / float(size)
        print('Epoch Accuracy', accuracy[epoch])
        print('Loss per Epoch', totalLoss/iters)
    print('Final Training  Accuracy: ', accuracy[max_epochs - 1])
    epoch_number = np.arange(0, max_epochs, 1)
    # Plot the training accuracy over epoch
    plt.figure()
    plt.plot(epoch_number, accuracy)
    plt.title('Training accuracy over Epoches')
    plt.xlabel('Number of Epoch')
    plt.ylabel('Accuracy')
    plt.show()

    #This calculates the test accuracy for each experiment
    testing_loader = DataLoader(dataset=DatasetTest(n), batch_size=batch_size, shuffle=shuffle)
    correct = 0
    nums = 0
    for i, data in enumerate(testing_loader, 0):
        inputs, labels = data
        inputs, labels = Variable(inputs), Variable(labels).long()
        y_pred = net(inputs)
        loss = criterion(y_pred, labels)
        optimizer.zero_grad()
        y_pred_np = y_pred.data.numpy()
        pred_np = np.argmax(y_pred_np, axis=1)
        label_np = labels.data.numpy().reshape(len(labels), 1)
        for j in range(y_pred_np.shape[0]):
            nums += 1
            if pred_np[j] == label_np[j, :]:
                correct += 1
    accuracyt = float(correct) / float(nums)
    print('Test Accuracy is ', accuracyt)

#Runs the experiment feed forward neural network without the normalization
runExperiment('NetExperimentOne', 0.001, 64, 0.9, True, 100, False)

#Runs the convolutional neural network with the normalization
#runExperiment('NetExperimentTwo', 0.001, 64, 0.9, True, 100, False)

normalize = True
#Runs the convolutional neural network with normalization and without parameter tuning
#runExperiment('NetExperimentTwo', 0.001, 64, 0.9,  True, 100, False)

#Runs the convolutional neural network with the normalization with parameter tuning
#runExperiment('NetExperimentTwo', 0.003, 64, 0.9,  True, 100, False)

#Runs the convolutional neural network with the normalization with parameter tuning
#runExperiment('NetExperimentTwo', 0.005, 64, 0.9,  True, 1, False)


















