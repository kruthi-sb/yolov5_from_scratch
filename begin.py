# Building YOLOv5s Backbone from scratch

import torch
import torch.nn as nn
import numpy as np

# utils
def same_padding(kernel_size, padding=None):
    """
    pads the i/p image to match the size of i/p and o/p image

    returns the padding size
    """
    if padding == None:
        padding = kernel_size//2 # floor div used for the convenience of using even/odd kernel_size

    # in theory, it is p=(kernel_size - 1)/2
    return padding


# YOLOv5s Backbone Layer Type 1: Conv 
class Conv(nn.Module):
    """
    Convolutional -> BatchNorm -> SiLU
    """

    default_act = nn.SiLU()  # default activation
    
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=None, groups=1, activation=True):

        super.__init__() # When initializing a derived class, you can use super() to call the __init__() method of the parent class.

        # define layers within
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, same_padding(kernel_size, padding), groups=groups, bias=False) # bias term is not there since we apply batch norm next, which  inherently subtracts out the bias

        self.bn = nn.BatchNorm2d(out_channels)

        self.activation = self.default_act if activation is True else activation if isinstance(activation, nn.Module) else nn.Identity() 

        # if act = true => use Silu, 
        # else if given a custom activation, like Relu, check if it is an instance of nn.Module and use it
        # else, use identity if none match.

    def forward(self, x):
        """
        x is the i/p tensor given.

        it applies the act(bn(conv)) and and returns o/p tensor.
        """
        return self.activation(self.bn(self.conv(x)))
    

# YOLOv5s Backbone Layer Type 2: C3 
class C3(nn.Module):
    def __init__(self ):
        super().__init__()
    






         

              



    





















# YOLOv5s Backbone Layer Type 2: C3
class C3(nn.Module):
    def __init__(self, in_channels, out_channels, n=1):
        super(C3, self).__init__()
        self.conv1 = Conv(in_channels, out_channels, 1)
        self.conv2 = Conv(out_channels, out_channels, 3)
        self.conv3 = Conv(out_channels, out_channels, 1)
        self.n = n

    def forward(self, x):
        for i in range(self.n):
            y = self.conv1(x)
            y = self.conv2(y)
            y = self.conv3(y)
            x = x + y
        return x
    
# YOLOv5s Backbone Layer Type 3: SPPF
class SPPF(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SPPF, self).__init__()
        self.conv1 = Conv(in_channels, out_channels, 1)
        self.conv2 = Conv(out_channels, out_channels, 3)
        self.conv3 = Conv(out_channels, out_channels, 1)
        self.conv4 = Conv(out_channels, out_channels, 3)
        self.conv5 = Conv(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.conv1(x)
        y = self.conv2(x)
        y = self.conv3(y)
        y = self.conv4(y)
        y = self.conv5(y)
        x = torch.cat((x, y), 1)
        return x
    
# YOLOv5s Backbone Layer Type 4: Concat
class Concat(nn.Module):
    def __init__(self, dimension=1):
        super(Concat, self).__init__()
        self.dimension = dimension

    def forward(self, *x):
        return torch.cat(x, self.dimension)
    
# YOLOv5s Backbone Layer Type 5: Detect
class Detect(nn.Module):
    def __init__(self, anchors, num_classes, stride):
        super(Detect, self).__init__()
        self.anchors = anchors
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.stride = stride
        self.scale_x_y = 1.0

    def forward(self, x):
        batch_size, channels, height, width = x.shape
        num_anchors = self.num_anchors
        num_classes = self.num_classes
        stride = self.stride
        scale_x_y = self.scale_x_y
        anchors = [(a[0] / stride, a[1] / stride) for a in self.anchors]

        # x: bs, 255, 20, 20
        x = x.view(batch_size, num_anchors, num_classes + 5, height, width).permute(0, 1, 3, 4, 2).contiguous()
        pred_x = torch.sigmoid(x[..., 0]) * scale_x_y - 0.5 * (scale_x_y - 1)
        pred_y = torch.sigmoid(x[..., 1]) * scale_x_y - 0.5 * (scale_x_y - 1)
        pred_w = x[..., 2]
        pred_h = x[..., 3]
        pred_conf = torch.sigmoid(x[..., 4])
        pred_cls = torch.sigmoid(x[..., 5:])

        grid_x = torch.arange(width, device=x.device).repeat(height, 1).view([1, 1, height, width]).float()
        grid_y = torch.arange(height, device=x.device).repeat(width, 1).t().view([1, 1, height, width]).float()
        anchor_w = torch.tensor(anchors, device=x.device).index_select(1, torch.tensor([0], device=x.device)).repeat([batch_size, 1]).repeat([1, 1, height, width])
        anchor_h = torch.tensor(anchors, device=x.device).index_select(1, torch.tensor([1], device=x.device)).repeat([batch_size, 1]).repeat([1, 1, height, width])

        pred_boxes = torch.zeros_like(x[..., :4])
        pred_boxes[..., 0] = pred_x.data + grid_x
        pred_boxes[..., 1] = pred_y.data + grid_y
        pred_boxes[..., 2] = torch.exp(pred_w.data) * anchor_w
        pred_boxes[..., 3] = torch.exp(pred_h.data) * anchor_h

        return torch.cat([pred_boxes.view(batch_size, -1, 4) * stride, pred_conf.view(batch_size, -1, 1), pred_cls.view(batch_size, -1, num_classes)], -1)
    
# YOLOv5s Backbone

class YOLOv5s(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv5s, self).__init__()
        self.num_classes = num_classes
        self.stride = [8, 16, 32]
        self.out_channels = [512, 256, 128]
        self.anchors = [[10, 13, 16, 30, 33, 23], [30, 61, 62, 45, 59, 119], [116, 90, 156, 198, 373, 326]]
        self.num_anchors = len(self.anchors)
        self.num_layers = len(self.anchors)
        self.num_outputs = self.num_anchors * (5 + num_classes)
        self.conv1 = Conv(3, 32, 3, 1)
        self.conv2 = Conv(32, 64, 3, 2)
        self.c3_1 = C3(64, 64)
        self.conv3 = Conv(64, 128, 3, 2)
        self.c3_2 = C3(128, 128)
        self.conv4 = Conv(128, 256, 3, 2)
        self.c3_3 = C3(256, 256)
        self.sppf = SPPF(256, 512)
        self.c3_4 = C3(512, 512)
        self.detect1 = Detect(self.anchors[0], num_classes, self.stride[0])
        self.conv5 = Conv(512, 256, 1)
        self.concat1 = Concat()
        self.c3_5 = C3(256 + 256, 256)
        self.detect2 = Detect(self.anchors[1], num_classes, self.stride[1])
        self.conv6 = Conv(256, 128, 1)
        self.concat2 = Concat()
        self.c3_6 = C3(128 + 128, 128)
        self.detect3 = Detect(self.anchors[2], num_classes, self.stride[2])

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.c3_1(x)
        x = self.conv3(x)
        x = self.c3_2(x)
        x = self.conv
        x = self.c3_3(x)
        x = self.sppf(x)
        x = self.c3_4(x)
        y1 = self.detect1(x)
        x = self.conv5(x)
        x = self.concat1(x)
        x = self.c3_5(x)
        y2 = self.detect2(x)
        x = self.conv6(x)
        x = self.concat2(x)
        x = self.c3_6(x)

        return y1, y2, self.detect3(x)
    
# YOLOv5s Backbone
def YOLOv5s(num_classes=80):
    return YOLOv5s(num_classes)

# Test
if __name__ == '__main__':
    model = YOLOv5s()
    x = torch.randn(1, 3, 640, 640)
    y1, y2, y3 = model(x)
    print(y1.shape, y2.shape, y3.shape)

    # Total params: 7,359,616
    print(sum(p.numel() for p in model.parameters()))

