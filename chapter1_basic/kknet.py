import torch
import os
import numpy as np
from time import time
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
import torchvision
import sys
sys.path.append("..")
import d2lzh_pytorch as d2l


torch.manual_seed(1)
torch.cuda.manual_seed(1)
np.random.seed(1)

# hyper parameters
num_epochs = 50
lr = 1e-3
weight_decay = 1e-4
batch_size = 16
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# data set
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)


# model
class LeNet(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=(3, 3), stride=1, padding=(1, 1))
        self.pool1 = nn.MaxPool2d(kernel_size=2)

        self.conv2 = nn.Conv2d(6, 16, kernel_size=(5, 5), stride=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2)

        self.conv3 = nn.Conv2d(16, 120, kernel_size=(5, 5), stride=1)

        self.fc1 = nn.Linear(in_features=120, out_features=84)
        self.fc2 = nn.Linear(in_features=84, out_features=num_classes)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool1(x)
        x = F.relu(self.conv2(x))
        x = self.pool2(x)
        x = F.relu(self.conv3(x))

        # x = x.view((x.shape[0], x.shape[1]))
        x = torch.flatten(x, start_dim=1)
        output = F.relu(self.fc1(x))
        output = self.fc2(output)
        return output


def evalation(model,dataloader):
    correct, total = 0,0
    with torch.no_grad():
        for x,y in dataloader:
            output = model(x.to(device)).cpu()
            correct += (torch.argmax(output,dim=1) == y).sum().item()
            total += x.shape[0]
    acc = correct/total
    return acc


#train
# net = LeNet(num_classes=10)
net = torchvision.models.resnet18()
hiden_dimen = net.fc.in_features
net.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
net.fc = nn.Linear(hiden_dimen,out_features=10)
net.to(device)
loss_func = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters(),lr=lr,weight_decay=weight_decay)

# tensorboard init
if not os.path.exists('runs'):
    os.mkdir('runs')
writer = SummaryWriter('runs')

print('start training ...')
for epoch in range(num_epochs):
    epoch += 1
    start_time = time()
    running_loss = []
    for i, data in enumerate(train_iter):
        print('\r', 'batch {}'.format(i), end='')
        x, y = data
        output = net(x.to(device))
        loss = loss_func(output,y.to(device))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running_loss.append(loss.item())

    print('\nevaluating ...')
    train_acc = evalation(net,train_iter)
    test_acc = evalation(net,test_iter)
    duration = time() - start_time
    loss_per_epoch = np.array(running_loss).mean()
    # print("time consumming: {:.2f}s".format(duration))
    print('epoch:{} loss={:.4f} train accuracy: {:.2f}% test accuracy : {:.2f}%, time: {:.2f}s'.format(
        epoch, loss_per_epoch, train_acc * 100, test_acc * 100, duration))
    writer.add_scalar('time duration', scalar_value=duration, global_step=epoch)
    writer.add_scalar('loss per epoch', loss_per_epoch, epoch)
    writer.add_scalar('train accuracy', train_acc, epoch)
    writer.add_scalar('test accuracy', test_acc, epoch)


## tensorboard 的使用
## tensorboard init 初始化，写入runs文件夹
# if not os.path.exists('runs'):
#     os.mkdir('runs')
# writer = SummaryWriter('runs')
##训练时加入到writer里面
# writer.add_scalar()
##
# (torch-tutorial) E:\work\torch-tutorial>cd chapter1_dataoperate
# (torch-tutorial) E:\work\torch-tutorial\chapter1_dataoperate>tensorboard -h
# 在terminal E:\work\torch-tutorial\chapter1_dataoperate>tensorboard --logdir runs
