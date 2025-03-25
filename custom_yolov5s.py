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
    
    def __init__(self, c1, c2, kernel_size=1, stride=1, padding=None, groups=1, activation=True):

        super().__init__() # When initializing a derived class, you can use super() to call the __init__() method of the parent class.

        # define layers within
        # c1 = in_channels
        # c2 = out_channels = no. of filters
        self.conv = nn.Conv2d(c1, c2, kernel_size, stride, same_padding(kernel_size, padding), groups=groups, bias=False) # bias term is not there since we apply batch norm next, which  inherently subtracts out the bias

        self.bn = nn.BatchNorm2d(c2)

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
    

# YOLOv5s Backbone Layer Type 2: BottleNeck
class Bottleneck(nn.Module):
    """
    c1 -> c_ -> c2, where c_ = c2*e, e for expansion factor (say, 0.5)

    x -> conv1 -> conv2 -> _ + x

    skip connection is added iff shortcut is True and c1==c2.
    o/p = x + conv2(conv1(x)), where '+' for skip connection.
    """
    def __init__(self, c1, c2, shortcut=True, groups=1, e=0.5 ):
        super().__init__()
        c_ = int(c2 * e) # is the no. of o/p channels for cv1.
        # => eg. c1 = 64, c2 = 128, e = 0.5, then c_ = 128/2 = 64 channels (hidden).

        # define layers
        self.cv1 = Conv(c1, c_, kernel_size=1, stride=1)
        self.cv2 = Conv(c_, c2, kernel_size=3, stride=1, groups=groups)

        self.add = shortcut and c1==c2 # whether T/F
        # adds skip connections iff c1==c2 & shortcut==T

    def forward(self, x):
        """
        x is the i/p tensor
        """
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
    

# YOLOv5s Backbone Layer Type 3: C3
class C3(nn.Module):
    """
    n = #bottleneck blocks inside the C3 module = depth
    c_ = o/p of first conv 
                         (c1 to c_)    (2*c_ to c2)
    x -> cv1 -> Bottleneck -> concat -> cv3 
    '--> cv2--------------------'
        (c1 to c_)

    """
    def __init__(self, c1, c2, n=1, shortcut=True, groups=1, e=0.5, backbone=True):
        super().__init__()
        c_ = int(c2 * e)  # hidden channels, e = width_multiple

        self.cv1 = Conv(c1, c_, kernel_size=1, stride=1) # 1x1 conv
        self.cv2 = Conv(c1, c_, 1, 1) #1x1 conv

        if backbone: # used in backbone part
            self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, groups, e=1.0) for _ in range(n)))
        else: # used in neck with just 2 convs
            self.m = nn.Sequential(
                *[nn.Sequential(
                    Conv(c1=c_, c2=c_, kernel_size=1, stride=1, padding=0),
                    Conv(c1=c_, c2=c_, kernel_size=3, stride=1, padding=1)
                ) for _ in range(n) #depth
                ]
            )
        self.cv3 = Conv(2*c_, c2, 1, 1) # 1x1 conv

    def forward(self, x):
        return self.cv3(torch.cat( (self.m(self.cv1(x)), self.cv2(x)),  dim=1))
    
# YOLOv5s Backbone Layer Type 4: SPPF
class SPPF(nn.Module):
    """
    cv1 -> [pool1->pool2->pool3] -> concat -> cv2
    '---------'------'------'--------'
    """
    def __init__(self, c1, c2):
        super().__init__()

        c_ = int(c1//2) # hidden channels

        self.cv1 = Conv(c1, c_, 1, 1) #1x1 conv
        self.pool = nn.MaxPool2d(kernel_size=5, stride=1, padding=2) 
        self.cv2 = Conv(c_*4, c2, 1, 1) #1x1 conv

    def forward(self, x):
        x_out = self.cv1(x)
        pool1 = self.pool(x_out)
        pool2 = self.pool(pool1)
        pool3 = self.pool(pool2)
        
        # concat -> cv2
        return self.cv2(torch.cat([x_out, pool1, pool2, pool3], dim=1))
    

#---------------------------   
# Build backbone, neck, head
class YOLOV5S(nn.Module):
    """
    i/p: image tensor (eg. torch.Size([1, 3, 640, 640]))
    o/p: feature maps extractored from backbone.

    NOTE: the no_of filters, i.e, c2 = 32 (not 64) in my implementation. So, the numbers are adjusted accordingly.
    YOLOv5 v6.0 backbone
    backbone:
        [from, number, module, args]
    [
        [-1, 1, Conv, [64, 6, 2, 2]], # 0-P1/2  --> layer 0, 'P' = feature map
        [-1, 1, Conv, [128, 3, 2]], # 1-P2/4
        [-1, 3, C3, [128]],
        [-1, 1, Conv, [256, 3, 2]], # 3-P3/8
        [-1, 6, C3, [256]],
        [-1, 1, Conv, [512, 3, 2]], # 5-P4/16
        [-1, 9, C3, [512]],
        [-1, 1, Conv, [1024, 3, 2]], # 7-P5/32
        [-1, 3, C3, [1024]],
        [-1, 1, SPPF, [1024, 5]], # 9  
    ]

    from: from which layer the module input comes from. Uses python syntax so -1 indicates prior layer.
    number: indicates the number of times a module repeats or how many repeats repeatable modules like C3 use
    args: module arguments (input channels inherited automatically)

    args for Conv: [c1, c2, kernel_size=1, stride=1, padding=None, groups=1, activation=True]
    args for C3:   [c1, c2, n=1, shortcut=True, groups=1, e=0.5]

    # head:
    nc = no. of classes = 80 for coco

    """

    def __init__(self, c2=32, nc = 80):
        super().__init__()
        self.backbone = nn.ModuleList()  # Holds submodules in a list. Initialize internal Module state.

        # add to the module list, the backbone layers in the form of list:
        self.backbone += [
            # 0
            #[3, 32, 6, 2, 2]
            
            Conv(c1=3, c2=c2, kernel_size=6, stride=2, padding=2), 

            # 1
            #[32, 64, 3, 2]
            Conv(c1=c2, c2=c2*2, kernel_size=3, stride=2),

            # 2
            #[64, 64, 1] 
            C3(c1=c2*2, c2=c2*2, n=1),

            # 3
            #[64, 128, 3, 2]
            Conv(c1=c2*2, c2=c2*4, kernel_size=3, stride=2),

            # 4
            #[128, 128, 2] 
            C3(c1=c2*4, c2=c2*4, n=2),

            # SKIP CONNECTION: from here, the output1 gets concatenated with o/p after resize2

            # 5
            #[128, 256, 3, 2]  
            Conv(c1=c2*4, c2=c2*8, kernel_size=3, stride=2),
             
            # 6
            #[256, 256, 3] 
            C3(c1=c2*8, c2=c2*8, n=3),

            # SKIP CONNECTION: from here, the output2 get concatenated with o/p after resize1

            # 7
            #[256, 512, 3, 2] 
            Conv(c1=c2*8, c2=c2*16, kernel_size=3, stride=2),

            # 8
            #[512, 512, 1]   
            C3(c1=c2*16, c2=c2*16, n=1),

            # 9
            # next will be SPPF: conv -> conv (1) -> maxpool (2) -> maxpool (3) -> maxpool (4) -> concat(1,2,3,4) -> conv -> conv -> Resize -> concat output2 -> and the remaining part of neck...

            # [512, 512, 5]
            SPPF(c1=c2*16, c2=c2*16)

            # ....
        ]

        self.neck = nn.ModuleList() # Initialize internal Module state

        # NOTE: While Ultralytics YOLOv5 uses nn.Upsample and nn.Concat (to facilitate versioning via .yaml files), in my implementation, I’ve used torch.nn.functional, and therefore, upsampling and concatenation are performed in the forward method.
         
        self.neck += [

            #-------- Upsampling:

            # o/p from SPPF goes to:

            # 10
            # [512, 256, 1, 1]
            Conv(c1=c2*16, c2=c2*8, kernel_size=1, stride=1, padding=0),

            # 11
            torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest']

            # 12
            # [-1, 6] -  models.common.Concat[1]

            # 13
            # [512, 256, 1, False]
            C3(c1=c2*16, c2=c2*8, e=0.25, n=2, backbone=False),

            # 14
            # [256, 128, 1, 1]
            Conv(c1=c2*8, c2=c2*4, kernel_size=1, stride=1, padding=0),

            # 15
            # torch.nn.modules.upsampling.Upsample    [None, 2, 'nearest'] 

            # 16
            # [-1, 4] - models.common.Concat[1]

            # 17
            # [256, 128, 1, False] 
            C3(c1=c2*8, c2=c2*4, e=0.25, n=2, backbone=False),

            #-------- Downsampling:
            # 18
            # [128, 128, 3, 2]  
            Conv(c1=c2*4, c2=c2*4, kernel_size=3, stride=2, padding=1),

            # 19
            # [-1, 14] - models.common.Concat[1]

            # 20
            # [256, 256, 1, False] 
            C3(c1=c2*8, c2=c2*8, e=0.5, n=2, backbone=False),

            # 21
            # [256, 256, 3, 2]
            Conv(c1=c2*8, c2=c2*8, kernel_size=3, stride=2, padding=1),

            # 22
            # [-1, 10] - models.common.Concat[1]

            # 23
            # [512, 512, 1, False]
            C3(c1=c2*16, c2=c2*16, e=0.5, n=2, backbone=False)
        ]

        # add self.head here - create instance of HEADS() class

    # make connections from [backbone <--> neck] and 
    # internal connections in neck from the NECK[upsampling <--> downsampling]

    def forward(self, x): # x is the input tensor

        # make sure that the input tensor has shape [#images_per_batch, 3 channels, width:some multiple of 32,  height:some multiple of 32]
        assert x.shape[2] % 32 == 0 and x.shape[3] % 32 == 0
        
        # store the outputs from backbone that should be concatenated with the layers of neck.
        backbone_connections = []

        # store the outputs from upsampling of neck that should be concatenated with the the layers of downsampling.
        neck_connections = []

        # store the end outputs after processing through backbone and neck
        neck_outputs = []

        # pass the input to the backbone
        for id, layer in enumerate(self.backbone):
            # pass to each layer sequencialy
            x = layer(x)

            # store o/ps from 4th and 6th layer:
            if id in [4,6]:
                backbone_connections.append(x)

        # pass o/p from backbone to neck
        for id, layer in enumerate(self.neck):
            # 


 

    """
    ...pending...
    add self.head 
    """
