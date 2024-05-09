import torch
import torch.nn as nn

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class MBConv1(nn.Module):
    def __init__(self, kernelSize, in_channels, out_channels, reduction_ratio):
        super(MBConv1, self).__init__()
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=kernelSize, stride=1, groups=int(in_channels/2), padding=int((kernelSize-1)/2))
        self.batch = nn.BatchNorm2d(in_channels)
        #activation function
        self.se = SEBlock(inputSize=in_channels, SE_reduce=reduction_ratio)
        self.decreaseChannels = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1)
        self.batch2 = nn.BatchNorm2d(out_channels)
    def forward(self,x):
        x = self.depthwise(x)
        x = self.batch(x)
        x = nn.SiLU()(x)
        x = self.se(x)
        x = self.decreaseChannels(x)
        x = self.batch2(x) 
        return x # smaller size, greater number of channels
    
class MBConv6(nn.Module):
    def __init__(self, kernelSize, in_channels, out_channels, stride, reduction_ratio):
        super(MBConv6, self).__init__()
        self.increaseChannels = nn.Conv2d(in_channels=in_channels, out_channels=in_channels*6, kernel_size=1)
        self.batch = nn.BatchNorm2d(in_channels * 6)
        self.depthwise = nn.Conv2d(in_channels=in_channels, out_channels=in_channels * 6, kernel_size=kernelSize, stride=stride, groups=in_channels, padding=kernelSize//2)
        #activation function
        self.se = SEBlock(inputSize=in_channels * 6, SE_reduce=reduction_ratio)
        self.decreaseChannels = nn.Conv2d(in_channels=in_channels*6, out_channels=out_channels, kernel_size=1)
        self.batch2 = nn.BatchNorm2d(out_channels)

        if(in_channels == out_channels and stride == 1):
            self.add = True
        else:
            self.add = False
    def forward(self,x):
        #out = self.increaseChannels(x) #increase channels by 6
        #out = self.batch(out)
        #out = Swish()(out)
        out = self.depthwise(x)
        out = self.batch(out)
        out = Swish()(out)
        out = self.se(out)
        out = self.decreaseChannels(out)
        out = self.batch2(out)

        if(self.add):
            return x + out
        else:
            return out # smaller size, greater number of channels
    















# https://iq.opengenus.org/efficientnet/
class SEBlock(nn.Module):
    def __init__(self, inputSize, SE_reduce):
        super(SEBlock, self).__init__()
        self.block = nn.Sequential(
            # Squeeze with avg pooling
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(inputSize, SE_reduce,1, bias=False),
            Swish(),
            # Excitation with linear layer
            nn.Conv2d(SE_reduce, inputSize,1, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        return x * self.block(x) # Inverted because it multiplies

stage1 = 32
stage2 = 16
stage3 = 24
stage4 = 40
stage5 = 80
stage6 = 112
stage7 = 192
stage8 = 320
stage9 = 1280
p_dropout = 0.2 #0.45
    
class EfficientNet(nn.Module):
    def __init__(self, SE_reduce, outputClasses):
        super(EfficientNet, self).__init__()
        self.SE_reduce = SE_reduce
        self.outputClasses = outputClasses

        # 32x32x3
        self.inputLayer = nn.Conv2d(in_channels=3, out_channels=stage1, kernel_size=3, stride=2, padding=int(1))
        # 16x16x32
        # Block 1
        self.block1 = MBConv1(in_channels=stage1,out_channels=stage2,kernelSize=3, reduction_ratio=self.SE_reduce)
        # 16x16x16
        # Block 2
        self.block2 = nn.Sequential(
            MBConv6(in_channels=stage2,out_channels=stage3,kernelSize=3,stride=2,reduction_ratio=self.SE_reduce),
            # 8x8x24
            MBConv6(in_channels=stage3, out_channels=stage3,stride=1, kernelSize=3,reduction_ratio=self.SE_reduce)  
            # 8x8x24
        )
        # 8x8x24
        # Block 3
        self.block3 = nn.Sequential(
            MBConv6(in_channels=stage3,out_channels=stage4,kernelSize=5,stride=2,reduction_ratio=self.SE_reduce),
            # 4x4x40
            MBConv6(in_channels=stage4, out_channels=stage4,stride=1,kernelSize=5,reduction_ratio=self.SE_reduce)  
            # 4x4x40
        )
        # 4x4x40
        # Block 4
        self.block4 = nn.Sequential (
            MBConv6(in_channels=stage4,out_channels=stage5,kernelSize=3,stride=1,reduction_ratio=self.SE_reduce),
            # 4x4x80
            MBConv6(in_channels=stage5, out_channels=stage5,stride=1,kernelSize=3,reduction_ratio=self.SE_reduce),
            # 4x4x80
            MBConv6(in_channels=stage5, out_channels=stage5,stride=1,kernelSize=3,reduction_ratio=self.SE_reduce)
            # 4x4x80
        )
        # 4x4x80
        # Block 5
        self.block5 = nn.Sequential(
            MBConv6(in_channels=stage5,out_channels=stage6,kernelSize=5,stride=2,reduction_ratio=self.SE_reduce),
            # 2x2x112
            MBConv6(in_channels=stage6,out_channels=stage6,kernelSize=5,stride=2,reduction_ratio=self.SE_reduce),
            # 1x1x112
            MBConv6(in_channels=stage6, out_channels=stage6,stride=1,kernelSize=5,reduction_ratio=self.SE_reduce)
            # 1x1x112
        )
        # 1x1x112
        # Block 6
        self.block6 = nn.Sequential(
            MBConv6(in_channels=stage6,out_channels=stage7,kernelSize=5,stride=2,reduction_ratio=self.SE_reduce),
            # 1x1x192
            MBConv6(in_channels=stage7, out_channels=stage7,stride=1,kernelSize=5,reduction_ratio=self.SE_reduce),
            # 1x1x192
            MBConv6(in_channels=stage7, out_channels=stage7,stride=1,kernelSize=5,reduction_ratio=self.SE_reduce),
            # 1x1x192
            MBConv6(in_channels=stage7, out_channels=stage7,stride=1,kernelSize=5,reduction_ratio=self.SE_reduce)
            # 1x1x192
        )
        # 1x1x192
        # Block 7
        self.block7 = MBConv6(in_channels=stage7,out_channels=stage8,kernelSize=3,stride=2,reduction_ratio=self.SE_reduce)
        # 1x1x320
        self.outputBlock = nn.Sequential(
            #nn.Conv2d(in_channels=stage8, out_channels=stage9, kernel_size=1),
            # 1x1x1280
            #nn.AvgPool2d(1),
            # 1x1x1280
            nn.Flatten(),
            #nn.Dropout(p_dropout),
            nn.Linear(in_features=stage8, out_features=70), # was 32
            nn.Dropout(p_dropout),
            nn.Linear(in_features=70, out_features=self.outputClasses),
            #nn.Sigmoid()
        )



        
    def forward(self, x):
        # 32x32x3
        # Input
        x = self.inputLayer(x)
        # 16x16x32
        # Block 1
        x = self.block1(x)
        # 16x16x16
        # Block 2
        x = self.block2(x)
        # 8x8x24
        # Block 3
        x = self.block3(x)
        # 4x4x40
        # Block 4
        x = self.block4(x)
        # 4x4x80
        # Block 5
        x = self.block5(x)
        # 1x1x112
        # Block 6
        x = self.block6(x)
        # 1x1x192
        # Block 7
        x = self.block7(x)
        # 1x1x320
        # Output
        x = self.outputBlock(x)
        return x



