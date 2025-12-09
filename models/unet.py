import torch
import torch.nn as nn

class ConvBnRelu(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size = 3, padding = 1, stride = 1, bias = True):
        super(ConvBnRelu, self).__init__()

        self.conv = nn.Conv2d(in_channels = in_channels,out_channels = out_channels,
                              kernel_size = kernel_size, padding = padding, stride = stride, bias = bias)
        self.bn = nn.BatchNorm2d(out_channels)
        self.nonlin = nn.ReLU(inplace=True)

    def forward(self, x:torch.tensor):
        out = x
        out = self.conv(out)
        out = self.bn(out)
        out = self.nonlin(out)

        return out
    
class DoubleConvBnRelu(nn.Module):
    def __init__(self,in_channels_1, out_channels_1, kernel_size_1,in_channels_2, out_channels_2, kernel_size_2):
        super(DoubleConvBnRelu, self).__init__()

        self.ConvBnRelu_1 = ConvBnRelu(in_channels = in_channels_1,out_channels = out_channels_1,kernel_size = kernel_size_1)
        self.ConvBnRelu_2 = ConvBnRelu(in_channels = in_channels_2,out_channels = out_channels_2,kernel_size = kernel_size_2)

    def forward(self,x:torch.Tensor):
        out = x

        out = self.ConvBnRelu_1(out)
        out = self.ConvBnRelu_2(out)

        return out

class Unet(nn.Module):
    def __init__(self, in_channels = 1, out_channels = 1):
        super(Unet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.encoderConv1_ = DoubleConvBnRelu(self.in_channels, 64, 3, 64, 64, 3)
        self.pool1_ = nn.MaxPool2d(kernel_size = 2)

        self.encoderConv2_ = DoubleConvBnRelu(64, 128, 3, 128, 128, 3)
        self.pool2_ = nn.MaxPool2d(kernel_size = 2)

        self.encoderConv3_ = DoubleConvBnRelu(128, 256, 3, 256, 256, 3)
        self.pool3_ = nn.MaxPool2d(kernel_size = 2)

        self.encoderConv4_ = DoubleConvBnRelu(256, 512, 3, 512, 512, 3)
        self.pool4_ = nn.MaxPool2d(kernel_size = 2)

        #bottleneck
        self.bottleneck_ = DoubleConvBnRelu(512, 1024, 3, 1024, 1024, 3)

        #Decoder
        self.upConv4_ = nn.ConvTranspose2d(1024, 512, 2, stride=2, padding=0, bias=True)
        self.decoderConv4_ = DoubleConvBnRelu(1024, 512, 3, 512, 512, 3)

        self.upConv3_ = nn.ConvTranspose2d(512, 256, 2, stride=2, padding=0, bias=True)
        self.decoderConv3_ = DoubleConvBnRelu(512, 256, 3, 256, 256, 3)

        self.upConv2_ = nn.ConvTranspose2d(256, 128, 2, stride=2, padding=0, bias=True)
        self.decoderConv2_ = DoubleConvBnRelu(256, 128, 3, 128, 128, 3)

        self.upConv1_ = nn.ConvTranspose2d(128, 64, 2, stride=2, padding=0, bias=True)
        self.decoderConv1_ = DoubleConvBnRelu(128, 64, 3, 64, 64, 3)

        #Header
        self.header_ = nn.Conv2d(in_channels=64, out_channels=self.out_channels, kernel_size=3, stride=1, padding=1)


    def forward(self, x:torch.Tensor):
        # Encoder Path
        enc1 = self.encoderConv1_(x)
        p1 = self.pool1_(enc1)

        enc2 = self.encoderConv2_(p1)
        p2 = self.pool2_(enc2)

        enc3 = self.encoderConv3_(p2)
        p3 = self.pool3_(enc3)

        enc4 = self.encoderConv4_(p3)
        p4 = self.pool4_(enc4)

        # Bottleneck
        bottleneck = self.bottleneck_(p4)

        # Decoder 4
        dec4 = self.upConv4_(bottleneck)
        # concat4_ 구현 (dim=1은 채널 방향)
        concat4 = torch.cat((dec4, enc4), dim=1) 
        dec4 = self.decoderConv4_(concat4)

        # Decoder 3
        dec3 = self.upConv3_(dec4)
        # concat3_ 구현
        concat3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoderConv3_(concat3)

        # Decoder 2
        dec2 = self.upConv2_(dec3)
        # concat2_ 구현
        concat2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoderConv2_(concat2)

        # Decoder 1
        dec1 = self.upConv1_(dec2)
        # concat1_ 구현
        concat1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoderConv1_(concat1)

        # Header
        out = self.header_(dec1)
        
        return out

