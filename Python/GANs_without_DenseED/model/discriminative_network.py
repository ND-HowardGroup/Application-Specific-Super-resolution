import torch
import torch.nn as nn
import torch.nn.functional as F

def flatten(input):
    return input.view(input.size(0), -1)

class discriminative_network(nn.Module):
    def __init__(self):
        super(discriminative_network, self).__init__()
        # conv > relu > conv > BN > relu > conv > BN > relu > conv > BN > relu > conv > BN > relu
        in_features = 4
        self.conv1 = nn.Conv2d(1,in_features*2,kernel_size=2, stride=1, padding=0, bias=False)
        # self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_features*2,in_features*2,kernel_size=3, stride=1, padding=0, bias=False)
        self.norm2 = nn.BatchNorm2d(in_features*2)
        # self.relu = nn.ReLU(inplace=True)
        self.conv3 = nn.Conv2d(in_features*2,in_features*3,kernel_size=3, stride=2, padding=0, bias=False)
        self.norm3 = nn.BatchNorm2d(in_features*3)
        # self.relu = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(in_features*3,in_features*4,kernel_size=3, stride=2, padding=0, bias=False)
        self.norm4 = nn.BatchNorm2d(in_features*4)
        # self.relu = nn.ReLU(inplace=True)
        self.conv5 = nn.Conv2d(in_features*4,in_features*5,kernel_size=3, stride=2, padding=0, bias=False)
        self.norm5 = nn.BatchNorm2d(in_features*5)
        # self.relu = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(in_features*5,in_features*6,kernel_size=3, stride=2, padding=0, bias=False)
        self.norm6 = nn.BatchNorm2d(in_features*6)
        # self.relu = nn.ReLU(inplace=True)
        self.fc1 = nn.Linear(864, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.act = nn.Sigmoid()
    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = self.norm2(x)   
        x = F.relu(self.conv3(x))
        x = self.norm3(x)          
        x = F.relu(self.conv4(x))
        x = self.norm4(x)   
        x = F.relu(self.conv5(x))
        x = self.norm5(x)   
        x = F.relu(self.conv6(x))
        x = self.norm6(x)   
        x = flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.act(self.fc4(x))
        # print(x.shape)
        return x


        
# net = discriminative_network()
# print(net)
# x = torch.Tensor(16, 4, 128, 128)
# y = net.forward(x)
# print(y.shape)