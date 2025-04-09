import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from scipy.interpolate import CubicSpline
from sklearn.cluster import DBSCAN
import torchvision.models as models
IMG_HEIGHT = 720
IMG_WIDTH = 1280
NUM_CLASSES = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RESA(nn.Module):
    def __init__(self, in_channels):
        super(RESA, self).__init__()
        self.conv = nn.Conv2d(in_channels, in_channels, 1, padding=0)
        self.bn = nn.BatchNorm2d(in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.refine_conv = nn.Conv2d(in_channels, in_channels, 3, padding=1, groups=in_channels)
        self.refine_bn = nn.BatchNorm2d(in_channels)
        self.shifts = [
            (1, 0), (2, 0), (3, 0),    
            (-1, 0), (-2, 0), (-3, 0), 
            (0, -1), (0, -2), (0, -3), 
            (0, 1), (0, 2), (0, 3)     
        ]
        
        distances = torch.tensor([(i**2 + j**2) for i, j in self.shifts], dtype=torch.float32)
        self.weights = torch.exp(-distances / (2 * 1.5**2)).to(device)

    def forward(self, x):
        contributions = []
        for (shift_h, shift_w), weight in zip(self.shifts, self.weights):
            shifted = torch.roll(x, shifts=(shift_h, shift_w), dims=(2, 3))
            contrib = self.relu(self.bn(self.conv(shifted)))
            contrib = self.relu(self.refine_bn(self.refine_conv(contrib)))
            contributions.append(contrib * weight)
        
        out = x + 0.5 * sum(contributions) / sum(self.weights)
        return out
    
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Decoder, self).__init__()
        self.Up1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 2, 3, padding=1),
            nn.BatchNorm2d(in_channels // 2),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
        self.Up2 = nn.Sequential(
            nn.Conv2d(in_channels // 2, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )

        self.smooth_conv = nn.Conv2d(out_channels, out_channels, 5, padding=2, groups=out_channels)
        self.smooth_bn = nn.BatchNorm2d(out_channels)
        
        self.thin1 = nn.Conv2d(in_channels // 2, in_channels // 2, 3, padding=1, groups=in_channels // 2)
        self.thin2 = nn.Conv2d(out_channels, out_channels, 3, padding=1, groups=out_channels)
        
        self.relu = nn.ReLU(inplace=True)  

    def forward(self, x):
        x = self.Up1(x)
        x = x + torch.sigmoid(self.thin1(x)) * x 
        x = self.Up2(x)
        x = x + torch.sigmoid(self.thin2(x)) * x 
        
        x = self.relu(self.smooth_bn(self.smooth_conv(x)))
        return x
    
class LaneNet(nn.Module):
    def __init__(self, num_classes=NUM_CLASSES, k_iterations=4):
        super(LaneNet, self).__init__()
        resnet = models.resnet34(weights=models.ResNet34_Weights.DEFAULT)
        self.encoder = nn.Sequential(
            resnet.conv1,
            resnet.bn1,
            resnet.relu,
            resnet.maxpool,
            resnet.layer1,
            resnet.layer2,
            resnet.layer3,
            resnet.layer4
        ) 

        self.resa_layers = nn.ModuleList([RESA(512) for _ in range(k_iterations)])
        self.decoder = Decoder(512, 256)
        self.seg_head = nn.Conv2d(256, num_classes, 1)

    def forward(self, x):
        x = self.encoder(x)
        for resa in self.resa_layers:
            x = resa(x)
        x = self.decoder(x)
        seg_out = self.seg_head(x)
        seg_out = F.interpolate(seg_out, size=(IMG_HEIGHT, IMG_WIDTH), mode='bilinear', align_corners=True)
        seg_out = F.avg_pool2d(seg_out, kernel_size=3, padding=1, stride=1)
        return seg_out