'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import torch
import math
import torch.nn as nn
from torch.nn.modules.utils import _triple, _pair
import torch.nn.functional as F

import softpool_cuda
from SoftPool import soft_pool3d, SoftPool3d



'''
===  S T A R T  O F  C L A S S  S Q U E E Z E _ A N D _ R E C U R S I O N ===

    [About]

        nn.Module class that created a SRTG block

    [Init Args]

        - planes: Integer for the number of channels used as input.
        - type : String for the recurrent cell types to be used. Defaults to `gru`.
        - layers: Integer for the number of layers in the recurrent sub-network. Defaults to 2.

    [Methods]

        - __init__ : Class initialiser
        - forward: Function for the main sequence of operation execution.

'''
class Squeeze_and_Recursion(nn.Module):

    def __init__(self, planes, type='gru', layers=2):
        super(Squeeze_and_Recursion, self).__init__()

        self.type = type
        planes = _pair(planes)

        if (self.type == 'gru'):
            self.op = nn.GRU(input_size = planes[0], hidden_size = planes[1], num_layers = layers)
        else :
            self.op = nn.LSTM(input_size = planes[0], hidden_size = planes[1], num_layers = layers)


    def forward(self, x):
        tensor_shape = list(x.size())
        # Perform GlobalAvgPool operation frame-wise (for non-vector inputs)
        if (tensor_shape[3] > 1 and tensor_shape[4] > 1):
            x_pool = F.avg_pool3d(x,kernel_size=(1,tensor_shape[3],tensor_shape[4]),stride=1)
        else:
            x_pool = x
        # Reshaping to tensor of size [batch, channels, frames]
        x_glob = x_pool.squeeze(-1).squeeze(-1)
        # [batch, channels, frames] -> [frames, batch, channels]
        x_glob= x_glob.permute(2,0,1)
        # use squeezed tensor as GRU input
        self.op.flatten_parameters()
        r_out, _ = self.op(x_glob)
        # [frames, batch, channels] -> [batch, channels, frames]
        r_out = r_out.permute(1,2,0)
        # [batch, channels, frames] -> [batch, channels, frames, 1, 1]
        x_glob = r_out.unsqueeze(-1).unsqueeze(-1)
        return x * x_glob.clone(), x_glob.clone()
'''
===  E N D  O F  C L A S S  S Q U E E Z E _ A N D _ R E C U R S I O N ===
'''


'''
===  S T A R T  O F  C L A S S  M T C O N V  ===

    [About]

        nn.Module for creating a MT-Convolution.

    [Init Args]

        - in_planes: Integer for the number of channels used as input.
        - out_planes: Integer for the number of channels of the output.
        - kernel_size: Integer or Tuple for the kernel size to be used.
        - rate: Float for the per channel usage between the short and long streams. Defaults to 0.5.
        - stride: Integer for the kernel stride. Defaults to 1.
        - padding: Integer for zero pad. Defaults to 1.
        - groups: Integer for the number of groups (i.e. sub-streams to be used). Defaults to 1.
        - bias: Boolean for using or not using a bias term.
        - no_lateral: Boolean for enabling lateral connections from the short and long streams. Defaults to False.
        - pool: String for the pooling type to be used (average or softpool). Defaults to `avg`.

    [Methods]

        - __init__ : Class initialiser
        - forward: Function for the main sequence of operation execution.
        - __get_downsample_stride__ : Staticmethod that returns the stride based on which the volume will be downsampled through the conv operation.
        - _initialise_weights: Function for initialising the convolutional kernels in the MTConv module.

'''
class MTConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, rate=0.5, stride=1, padding=0, groups=1, bias=False, no_lateral=False, pool='avg'):

        super(MTConv, self).__init__()

        self.no_lateral = no_lateral
        self.pool = pool

        # Store short-long ratio
        self.r = rate

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        ####### C H A N N E L   C A L C U L A T I O N S #######
        #
        #                [SHORT]                        [LONG]
        #         |--------s_1--------|                  l_1
        #   (short2short)       (short2long)          (long2long)
        #        s_2               l_2*r              (l_2*(1-r))
        #         |                  |                     |
        #         |                  |                     |
        #         |                  --------------------[cat]
        #         |                                        |
        #         |                                        |
        #         |                                       l_2
        #      [SHORT]                                   [LONG]
        #

        groups_short = math.floor(groups * self.r) if math.floor(groups * self.r) > 1 else 1

        groups_long = groups - groups_short if groups - groups_short > 1 else 1

        self.in_planes_short = math.floor((int(in_planes * self.r))/groups_short)*groups_short
        self.out_planes_short = int(out_planes* self.r)

        self.in_planes_long = int(in_planes - self.in_planes_short)
        self.out_planes_long = int(out_planes - self.out_planes_short)

        self.relu = nn.ReLU(inplace=True)

        self.conv_short = nn.Conv3d(self.in_planes_short, self.out_planes_short, kernel_size,stride=stride, padding=padding, groups=groups_short, bias=False)
        self.conv_long = nn.Conv3d(self.in_planes_long, self.out_planes_long, kernel_size,stride=stride, padding=padding, groups=groups_long, bias=False)

        if not self.no_lateral:
            self.conv_short2long = nn.Conv3d(self.out_planes_short, self.out_planes_long, kernel_size=1, stride=1, padding=0, bias=False)

        if (groups>1):
            self.norm_short = nn.GroupNorm(groups_short,self.out_planes_short)
            self.norm_long = nn.GroupNorm(groups_long,self.out_planes_long)

        else:
            self.norm_short = nn.BatchNorm3d(self.out_planes_short)
            self.norm_long = nn.BatchNorm3d(self.out_planes_long)


    def forward(self, x):

        # Split volume
        x_short,x_long = x

        x_short_id = x_short

        x_short_id = x_short
        x_long_id = x_long

        # Short
        x_short = self.conv_short(x_short)

        # Long
        x_long = self.conv_long(x_long)

        if not self.no_lateral:

            _, cs, t_short, h_short, w_short = x_short.size()
            _, cl, t_long, h_long, w_long = x_long.size()

            if self.pool == 'soft':
                x_short2long = F.adaptive_max_pool3d(x_short,(x_short.size()[-3],x_short.size()[-2],x_short.size()[-1]))
                '''
                if (x_short.size()[-2]%2 != 0 or x_short.size()[-1]%2 != 0):
                    padding = (1,0,1,0,0,0) # pad last dim by (0, 1) and 2nd to last by (0, 1)
                    x_short2long = F.pad(x_short, padding, 'replicate')
                    x_short2long = soft_pool3d(x_short2long,kernel_size=(1,2,2),stride=(1,2,2))
                else:
                    x_short2long = soft_pool3d(x_short,kernel_size=(1,2,2),stride=(1,2,2))
                '''
            else:
                x_short2long = F.avg_pool3d(x_short,kernel_size=(1,2,2),stride=(1,2,2))


            if (list(x_short2long.size())[2:] != list(x_long[0].size())[2:]):
                t,h,w = list(x_long.size())[2:]
                x_short2long = F.interpolate(x_short2long,size=(t,h,w),mode='trilinear')

            x_short2long = self.conv_short2long(x_short2long)


        x_long = self.norm_long(x_long)
        x_short = self.norm_short(x_short)

        if not self.no_lateral:
            x_short2long = self.norm_long(x_short2long)

            x_long = torch.add(x_long,x_short2long)

        return (x_short,x_long)

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)

    def _initialise_weights(self):

        bns = [self.norm_short,self.norm_long]
        convs = [self.conv_short,self.conv_long]
        if not self.no_lateral:
            convs.append(self.conv_short2long)

        for m in convs:
            nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

        if bns is not None:
            for m in bns:
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
'''
===  E N D  O F  C L A S S  M T C O N V ===
'''



'''
===  S T A R T  O F  C L A S S  M T B L O C K ===

    [About]

        nn.Module for creating a MTBlock.

    [Init Args]

        - in_planes: Integer for the number of channels used as input.
        - planes: Integer for the number of channels of the output (if expansion == 1).
        - stride: Integer for the kernel stride. Defaults to 1.
        - downsample: nn.Module in the case that downsampling is to be used for the residual connection.
        Defaults to None.
        - groups: Integer for the number of groups (i.e. sub-streams to be used). Defaults to 1.
        - base_width: Used to define the width of the bottleneck. Defaults to 64.
        - rate: Float for the per channel usage between the short and long streams. Defaults to 0.875.

    [Methods]

        - __init__ : Class initialiser
        - forward: Function for the main sequence of operation execution.

'''
class MTBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1, base_width=64, rate=.875):


        super(MTBlock, self).__init__()

        width = int(planes * (base_width / 64.) / groups)

        width_ix = width


        # 1x1x1
        self.conv1_1 = MTConv(
                        in_planes=inplanes,
                        out_planes=width_ix,
                        kernel_size=1,
                        rate=rate)

        self.conv1_2 = MTConv(
                        in_planes=width_ix,
                        out_planes=inplanes,
                        kernel_size=1,
                        rate=rate)

        # Initial conv block
        if downsample is not None:
            self.conv2 = MTConv(
                            in_planes=inplanes,
                            out_planes=width,
                            kernel_size=3,
                            rate=rate,
                            stride=stride,
                            padding=1,
                            groups=groups)

            self.conv3 = MTConv(
                            in_planes=width,
                            out_planes=planes*self.expansion,
                            kernel_size=1,
                            rate=rate,
                            groups=groups)

        else:
            self.conv2 = MTConv(
                            in_planes=inplanes,
                            out_planes=width,
                            kernel_size=3,
                            rate=rate,
                            stride=stride,
                            padding=1,
                            groups=groups)

            self.conv3 = MTConv(
                            in_planes=width,
                            out_planes=planes*self.expansion,
                            kernel_size=(1,3,3),
                            rate=rate,
                            padding=(0,1,1),
                            groups=groups)

        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

        self.sr_short = Squeeze_and_Recursion(planes=int(planes*self.expansion * rate))
        self.sr_long = Squeeze_and_Recursion(planes=int(planes*self.expansion * (1-rate)))

        groups_short = math.floor(groups * rate) if math.floor(groups * rate) > 1 else 1
        planes_short = math.floor((int(planes * rate))/groups_short)*groups_short
        planes_long = int(planes - planes_short)



    def forward(self, x):
        identity_short,identity_long = x

        out_short,out_long = self.conv1_1(x)
        self.relu(out_short)
        self.relu(out_long)

        out_short,out_long = self.conv1_2((out_short,out_long))
        self.relu(out_short)
        self.relu(out_long)

        out_short,out_long = self.conv2((out_short,out_long))
        self.relu(out_short)
        self.relu(out_long)

        out_short,out_long = self.conv3((out_short,out_long))

        if self.downsample is not None:
            if (identity_long.size()[3]<3 and identity_long.size()[3]<3):
                identity_long = F.interpolate(identity_long,size=(identity_long.size()[2],3,3),mode='trilinear')
            identity_short,identity_long = self.downsample((identity_short,identity_long))

        # Ensure that sizes match
        if (list(out_short.size())[2:] != list(identity_short.size())[2:]):
            t,h,w = list(out_short.size())[2:]
            identity_short = F.interpolate(identity_short,size=(t,h,w),mode='trilinear')
        if (list(out_long.size())[2:] != list(identity_long.size())[2:]):
            t,h,w = list(out_long.size())[2:]
            identity_long = F.interpolate(identity_long,size=(t,h,w),mode='trilinear')


        out_short,_ = self.sr_short(out_short)
        out_long,_ = self.sr_long(out_long)

        out_short += identity_short
        out_long += identity_long
        self.relu(out_short)
        self.relu(out_long)

        return (out_short,out_long)
'''
===  E N D  O F  C L A S S  M T B L O C K ===
'''


'''
===  S T A R T  O F  C L A S S  B A S I C S T E M ===

    [About]

        nn.Sequential Class for the initial 3D convolution.

    [Init Args]

        - None

    [Methods]

        - __init__ : Class initialiser
'''
class BasicStem(nn.Sequential):
    def __init__(self):
        super(BasicStem, self).__init__(
            nn.Conv3d(3, 64, kernel_size=(3, 7, 7), stride=(1, 2, 2),
                      padding=(1, 3, 3), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
'''
===  E N D  O F  C L A S S  B A S I C S T E M ===
'''


'''
===  S T A R T  O F  C L A S S  M T N E T ===

    [About]

        nn.Module for creating MTnet variants.

    [Init Args]

        - block: nn.Module used as building block.
        - layers: List of Integers specifying the number of blocks per layer.
        - num_classes: Integer for the dimension of the final FC layer. Defaults to 400.
        - rate: Float for the per channel usage between the short and long streams. Defaults to 0.875.
        - width_per_group: Integer for the number of channels that each group/sub-stream will have.
        - zero_init_residual: Boolean for zero init bottleneck residual BN. Defaults to False.

    [Methods]

        - __init__ : Class initialiser
        - forward: Function for performing the main sequence of operations.
        - _make_layer: Function for creating a sequence (nn.Sequential) of layers.
        - _initialise_weights: Function for weight initialisation.
'''
class MTnet(nn.Module):

    def __init__(self, block, layers, num_classes=400, groups=1, width_per_group=64,
                 zero_init_residual=False, r=.875):

        super(MTnet, self).__init__()
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group
        self.rate = r

        self.init_planes_short = int(self.inplanes * self.rate)
        self.init_planes_long = self.inplanes - self.init_planes_short


        self.pool = nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))

        self.stem = BasicStem()

        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        self._initialise_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, MTBlock):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)
        if (x.size()[-2]%2 != 0 or x.size()[-1]%2 != 0):
            padding = (1,0,1,0,0,0) # pad last dim by (0, 1) and 2nd to last by (0, 1)
            x = F.pad(x, padding, 'replicate')
        x = self.pool(x)

        # Short/Long path creation
        x_short,x_long = torch.split(x,[self.init_planes_short,self.init_planes_long],dim=1)
        x_long = F.avg_pool3d(x_long, kernel_size=(1,2,2),stride=(1,2,2))

        # group together
        x = (x_short, x_long.clone())
        x = self.layer1(x)

        x = self.layer2(x)

        x = self.layer3(x)

        x = self.layer4(x)

        x_short, x_long  = x

        _, _, t_short, h_short, w_short = x_short.size()
        _, _, t_long, h_long, w_long = x_long.size()

        x_short = self.avgpool(x_short)
        x_long = self.avgpool(x_long)

        x = torch.cat([x_short,x_long],dim=1)

        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = MTConv.get_downsample_stride(stride)

            downsample = MTConv(
                            in_planes=self.inplanes,
                            out_planes=planes*block.expansion,
                            kernel_size=1,
                            rate=self.rate,
                            stride=ds_stride,
                            padding=0,
                            bias=False,
                            no_lateral=True)

        layers = []
        layers.append(block(inplanes=self.inplanes, planes=planes, stride=stride, downsample=downsample, groups=self.groups, base_width=self.base_width, rate=self.rate))

        self.inplanes = planes * block.expansion

        for i in range(1, blocks):
            layers.append(block(inplanes=self.inplanes, planes=planes, groups=self.groups, base_width=self.base_width, rate=self.rate))

        return nn.Sequential(*layers)

    def _initialise_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, MTConv):
                m._initialise_weights()
'''
===  E N D  O F  C L A S S  M T N E T ===
'''


'''
---  S T A R T  O F  N E T W O R K  C R E A T I O N  F U N C T I O N S ---
    [About]
        All below functions deal with the creation of networks. Networks are specified based on their
        function names.
'''
def MTNet_32(**kwargs):
    return MTnet(block=MTBlock, layers=[2, 2, 2, 2], **kwargs)

def MTNet_48(**kwargs):
    return MTnet(block=MTBlock, layers=[2, 2, 5, 3], **kwargs)

def MTNet_64(**kwargs):
    return MTnet(block=MTBlock, layers=[3, 4, 6, 3], **kwargs)

def MTNet_132(**kwargs):
    return MTnet(block=MTBlock, layers=[3, 4, 23, 3], **kwargs)

def MTNet_200(**kwargs):
    return MTnet(block=MTBlock, layers=[3, 8, 36, 3], **kwargs)

def MTNet_264(**kwargs):
    return MTnet(block=MTBlock, layers=[3, 24, 36, 3], **kwargs)


def MTNet_32_g8(**kwargs):
    return MTnet(block=MTBlock, groups=8, layers=[2, 2, 2, 2], **kwargs)

def MTNet_48_g8(**kwargs):
    return MTnet(block=MTBlock, groups=8, layers=[2, 2, 5, 3], **kwargs)

def MTNet_64_g8(**kwargs):
    return MTnet(block=MTBlock, groups=8, layers=[3, 4, 6, 3], **kwargs)

def MTNet_132_g8(**kwargs):
    return MTnet(block=MTBlock, groups=8, layers=[3, 4, 23, 3], **kwargs)

'''
---  E N D  O F  N E T W O R K  C R E A T I O N  F U N C T I O N S ---
'''
