import torch
import torch.nn as nn


class ResNet_block(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super(ResNet_block, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1,stride=stride)
        self.conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, padding=1,stride=1)
        self.bn = nn.BatchNorm2d(num_features=out_channels)
        self.drop = nn.Dropout(p=0.3)

        self.bypass = nn.Sequential()
        if stride != 1:
            self.bypass = nn.Sequential(
                nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(num_features=out_channels)
            )

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.bn(x)
        x = self.drop(x)
        x = self.conv2(x)
        x = self.bn(x)
        x += self.bypass(input)
        x = torch.relu(x)
        x = self.drop(x)

        return x


# Defining the network
class ResNet18_32x32(nn.Module):
    def __init__(self, block):
        super(ResNet18_32x32, self).__init__()
        # First Conv layer
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=1)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=1,stride=1)
        # ResNet Blocks
        self.layer1 = self.__make_layer(block=block,in_channels=64,out_channels=64,stride=1)
        self.layer2 = self.__make_layer(block=block,in_channels=64,out_channels=128,stride=2)
        self.layer3 = self.__make_layer(block=block,in_channels=128,out_channels=256,stride=2)
        self.layer4 = self.__make_layer(block=block,in_channels=256,out_channels=512,stride=2)
    
        self.avePool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(in_features=512,out_features=3)

    def __make_layer(self, block, in_channels, out_channels, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        layers.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avePool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    
    # Defining the network
class ResNet18_64x64(nn.Module):
    def __init__(self, block):
        super(ResNet18_64x64, self).__init__()
        # First Conv layer
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=1)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=1,stride=1)
        # ResNet Blocks
        self.layer1 = self.__make_layer(block=block,in_channels=64,out_channels=64,stride=1)
        self.layer2 = self.__make_layer(block=block,in_channels=64,out_channels=128,stride=2)
        self.layer3 = self.__make_layer(block=block,in_channels=128,out_channels=256,stride=2)
        self.layer4 = self.__make_layer(block=block,in_channels=256,out_channels=512,stride=2)
    
        self.avePool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(in_features=512,out_features=3)

    def __make_layer(self, block, in_channels, out_channels, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        layers.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avePool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    
        # Defining the network
class ResNet18_96x96(nn.Module):
    def __init__(self, block):
        super(ResNet18_96x96, self).__init__()
        # First Conv layer
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=1)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.pool = nn.MaxPool2d(kernel_size=1,stride=1)
        # ResNet Blocks
        self.layer1 = self.__make_layer(block=block,in_channels=64,out_channels=64,stride=2)
        self.layer2 = self.__make_layer(block=block,in_channels=64,out_channels=128,stride=2)
        self.layer3 = self.__make_layer(block=block,in_channels=128,out_channels=256,stride=2)
        self.layer4 = self.__make_layer(block=block,in_channels=256,out_channels=512,stride=2)
    
        self.avePool = nn.AvgPool2d(kernel_size=4)
        self.fc = nn.Linear(in_features=512,out_features=3)

    def __make_layer(self, block, in_channels, out_channels, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        layers.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.pool(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avePool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x

class ResNet34_224x224(nn.Module):
    def __init__(self, block):
        super(ResNet34_224x224, self).__init__()
        # First Conv layer
        self.conv1 = nn.Conv2d(in_channels=3,out_channels=64,kernel_size=7,stride=2, padding=3)
        self.bn = nn.BatchNorm2d(num_features=64)
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(kernel_size=3,stride=2,padding=1)
        # ResNet Blocks
        self.layer1 = nn.Sequential(
            self.__make_layer(block=block,in_channels=64,out_channels=64,stride=1),
            self.__make_layer(block=block,in_channels=64,out_channels=64,stride=1),
            self.__make_layer(block=block,in_channels=64,out_channels=128,stride=2)
        )
        self.layer2 = nn.Sequential(
            self.__make_layer(block=block,in_channels=128,out_channels=128,stride=1),
            self.__make_layer(block=block,in_channels=128,out_channels=128,stride=1),
            self.__make_layer(block=block,in_channels=128,out_channels=128,stride=1),
            self.__make_layer(block=block,in_channels=128,out_channels=256,stride=2)
        )
        self.layer3 = nn.Sequential(
            self.__make_layer(block=block,in_channels=256,out_channels=256,stride=1),
            self.__make_layer(block=block,in_channels=256,out_channels=256,stride=1),
            self.__make_layer(block=block,in_channels=256,out_channels=256,stride=1),
            self.__make_layer(block=block,in_channels=256,out_channels=256,stride=1),
            self.__make_layer(block=block,in_channels=256,out_channels=256,stride=1),
            self.__make_layer(block=block,in_channels=256,out_channels=512,stride=2)
        )
        self.layer4 = nn.Sequential(
            self.__make_layer(block=block,in_channels=512,out_channels=512,stride=1),
            self.__make_layer(block=block,in_channels=512,out_channels=512,stride=1),
            self.__make_layer(block=block,in_channels=512,out_channels=512,stride=1)
        )

    
        self.avePool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(in_features=512,out_features=3)

    def __make_layer(self, block, in_channels, out_channels, stride):
        layers = []
        layers.append(block(in_channels, out_channels, stride))
        #layers.append(block(out_channels, out_channels, 1))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x) 
        x = self.pool(x)


        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avePool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x