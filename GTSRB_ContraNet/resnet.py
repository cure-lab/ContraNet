'''ResNet in PyTorch.

For Pre-activation ResNet, see 'preact_resnet.py'.

Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.quantization import QuantStub, DeQuantStub
from thop import profile

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        # replace torch add by quantized float functional for quantization
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # out += self.shortcut(x)
        out = self.skip_add.add(out, self.shortcut(x))
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes)
            )
        # replace torch add by quantized float functional for quantization
        self.skip_add = nn.quantized.FloatFunctional()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        # out += self.shortcut(x)
        out = self.skip_add.add(out, self.shortcut(x))
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=43):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.dequant(out)

        return out

    # Fuse Conv+BN and Conv+BN+Relu modules prior to quantization
    # This operation does not change the numerics
    def fuse(self):

        # fuse the first conv+bn+relu layer
        torch.quantization.fuse_modules(self, ['conv1', 'bn1'], inplace=True)

        # fuse the 4 layers
        for i in range(4):
            print ('fusing layer %d' % (i+1))
            seq_module = self.__dict__['_modules']['layer'+str(i+1)]
            for name, block_module in seq_module.__dict__['_modules'].items():
                torch.quantization.fuse_modules(block_module, [['conv1', 'bn1'], ['conv2', 'bn2']], inplace=True)
                if isinstance(block_module, Bottleneck):
                    # print ('bottleneck module\n')
                    torch.quantization.fuse_modules(block_module, [['conv3', 'bn3']], inplace=True)
                # if block_module.__dict__['_modules']['shortcut'].children() != None:
                if (len(list(block_module.__dict__['_modules']['shortcut'].children()))) != 0:
                    # print (sm)
                    torch.quantization.fuse_modules(block_module.__dict__['_modules']['shortcut'], ['0', '1'], inplace=True)
        
        # remove the empty shortcut layer in Basic block 
        for name, module in self.named_children():
            if isinstance(module, nn.Sequential):
                for m in module.children():
                    if (m.shortcut.__len__() == 0): # if the shortcut is empty
                        m.shortcut = nn.Identity()
    def profiling(self):
        img = torch.ones(1, 3, 32, 32).to(device)
        total_ops, total_params = profile(self, inputs=(img, ), verbose=False)
        print("params %.2f | flops %.2f" % (total_params / (1000 ** 2), total_ops / (1000 ** 3)))
        
        return total_ops, total_params


def ResNet10():
    return ResNet(BasicBlock, [1, 1, 1, 1])

def ResNet18():
    return ResNet(BasicBlock, [2,2,2,2])

def ResNet34():
    return ResNet(BasicBlock, [3,4,6,3])

def ResNet50():
    return ResNet(Bottleneck, [3,4,6,3])

def ResNet101():
    return ResNet(Bottleneck, [3,4,23,3])

def ResNet152():
    return ResNet(Bottleneck, [3,8,36,3])

def ResNet200():
    return ResNet(Bottleneck, [3, 24, 36, 3])

class ResNetAPP(nn.Module):
    def __init__(self, block, num_blocks, size=1, num_classes=43):
        super(ResNetAPP, self).__init__()
        self.in_planes = 32

        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layer1 = self._make_layer(block, int(32*size), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(64*size), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(128*size), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(256*size), num_blocks[3], stride=2)
        self.linear = nn.Linear(int(256*size)*block.expansion, num_classes)
        self.quant = QuantStub()
        self.dequant = DeQuantStub()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.quant(x)
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = self.dequant(out)

        return out
    def fuse(self):

        # fuse the first conv+bn+relu layer
        torch.quantization.fuse_modules(self, ['conv1', 'bn1'], inplace=True)

        # fuse the 4 layers
        for i in range(4):
            print ('fusing layer %d' % (i+1))
            seq_module = self.__dict__['_modules']['layer'+str(i+1)]
            for name, block_module in seq_module.__dict__['_modules'].items():
                torch.quantization.fuse_modules(block_module, [['conv1', 'bn1'], ['conv2', 'bn2']], inplace=True)
                if isinstance(block_module, Bottleneck):
                    # print ('bottleneck module\n')
                    torch.quantization.fuse_modules(block_module, [['conv3', 'bn3']], inplace=True)
                # if block_module.__dict__['_modules']['shortcut'].children() != None:
                if (len(list(block_module.__dict__['_modules']['shortcut'].children()))) != 0:
                    # print (sm)
                    torch.quantization.fuse_modules(block_module.__dict__['_modules']['shortcut'], ['0', '1'], inplace=True)
        
        # remove the empty shortcut layer in Basic block 
        for name, module in self.named_children():
            if isinstance(module, nn.Sequential):
                for m in module.children():
                    if (m.shortcut.__len__() == 0): # if the shortcut is empty
                        m.shortcut = nn.Identity()
                        
    def profiling(self):
        img = torch.ones(1, 3, 32, 32).to(device)
        total_ops, total_params = profile(self, inputs=(img, ), verbose=False)
        # print("params %.2f M| flops %.2f G" % (total_params / (1000 ** 2), total_ops / (1000 ** 3)))
        
        return total_ops, total_params


def ResNet10APP(size, num_classes):
    return ResNetAPP(BasicBlock, [1, 1, 1, 1], size, num_classes)


def test():
    net = ResNet152()
    # y = net(torch.randn(1,3,32,32))
    # print(y.size())

    # net.eval()
    # # for k, v in net.state_dict().items():
    #     # print (k, v.shape)

    # # fusing
    # print ("net before fusing:")
    # print (net)

    # net.fuse_model()
    # print ('---------------')
    # print ("net after fusing:")
    # print (net)

    # for k, v in net.state_dict().items():
        # print (k, v.shape)
    # net = ResNet10APP(size=1, num_classes=5)
    y = net(torch.randn(1,3,32,32))
    print(y.size())
   
    net.profiling()

if __name__ == "__main__":
    test()
