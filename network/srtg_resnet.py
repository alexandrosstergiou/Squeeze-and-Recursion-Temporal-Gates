'''
---  I M P O R T  S T A T E M E N T S  ---
'''
import torch
import math
import torch.nn as nn
from torch.nn.modules.utils import _triple
import torch.nn.functional as F


'''
===  S T A R T  O F  C L A S S  C O N V 3 D S I M P L E ===

    [About]

        nn.Sequential class for creating a 3D Convolution operation used as a building block for the 3D CNN.

    [Init Args]

        - in_planes: Integer for the number of channels used as input.
        - out_planes: Integer for the number of channels of the output.
        - midplanes: None. Only used to ensure continouity with class `Conv2Plus1D`.
        - stride: Integer for the kernel stride. Defaults to 1.
        - padding: integer for zero pad. Defaults to 1.

    [Methods]

        - __init__ : Class initialiser
        - __get_downsample_stride__ : Staticmethod that returns the stride based on which the volume will be downsampled through the conv operation

'''
class Conv3DSimple(nn.Sequential):
    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DSimple, self).__init__(
            nn.Conv3d(in_planes, out_planes, kernel_size=(3, 3, 3),
                      stride=(stride, stride, stride), padding=(padding, padding, padding),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)
'''
===  E N D  O F  C L A S S  C O N V 3 D S I M P L E ===
'''


'''
===  S T A R T  O F  C L A S S  C O N V 2 P L U S 1 D ===

    [About]

        nn.Sequential class for creating a (2+1)D Convolution operation used as a building block for the 3D CNN.

    [Init Args]

        - in_planes: Integer for the number of channels used as input.
        - out_planes: Integer for the number of channels of the output.
        - midplanes: Integer for the number of intermediate channels as calculated by the (2+1)D operation.
        - stride: Integer for the kernel stride. Defaults to 1.
        - padding: integer for zero pad. Defaults to 1.

    [Methods]

        - __init__ : Class initialiser
        - __get_downsample_stride__ : Staticmethod that returns the stride based on which the volume will be downsampled through the conv operation

'''
class Conv2Plus1D(nn.Sequential):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes,
                 stride=1,
                 padding=1):
        super(Conv2Plus1D, self).__init__(
            nn.Conv3d(in_planes, midplanes, kernel_size=(1, 3, 3),
                      stride=(1, stride, stride), padding=(0, padding, padding),
                      bias=False),
            nn.BatchNorm3d(midplanes),
            nn.ReLU(inplace=True),
            nn.Conv3d(midplanes, out_planes, kernel_size=(3, 1, 1),
                      stride=(stride, 1, 1), padding=(padding, 0, 0),
                      bias=False))

    @staticmethod
    def get_downsample_stride(stride):
        return (stride, stride, stride)
'''
===  E N D  O F  C L A S S  C O N V 2 P L U S 1 D ===
'''


'''
===  S T A R T  O F  C L A S S  S Q U E E Z E _ A N D _ R E C U R S I O N ===

    [About]

        nn.Module class that created a SRTG block

    [Init Args]

        - planes: Integer for the number of channels used as input.
        - layers: Integer for the number of layers in the recurrent sub-network. Defaults to 2.
        - gates: Boolean for using Temporal Gates. Defaults to False.

    [Methods]

        - __init__ : Class initialiser
        - soft_nnc : Function for checking the cyclic consistency based on the soft neared neighbour
        of two embedding/feature spaces. It will vectorise both volume to temporal sequences and calculate
        their element-wise euclidian distance. The end result is returned as a matrix of True or False
        (as 1s and 0s) in order to only fused together the specific examples in the batch.
        - euclidian_distance: DEPRECATED(!) Function for calculating euclidean distance. Only included as a reference - should not be used.
        - intersect1d: Function from Mask R-CNN that can find intersecting points between two pytorch tensors.
        Used to compare the two nearest nighbour arrays.
        - forward: Function for the main sequence of operation execution.

'''
class Squeeze_and_Recursion(nn.Module):

    def __init__(self, planes, layers=2, gates=False):
        super(Squeeze_and_Recursion, self).__init__()

        self.lstm = nn.LSTM(input_size = planes, hidden_size = planes, num_layers = layers)
        self.gate = gates


    def soft_nnc(self,embeddings1,embeddings2):

        # Assume inputs shapes of (batch x) x channels x frames x height x width
        dims_1 = embeddings1.shape
        dims_2 = embeddings2.shape

        # Pooling Height and width to create frame-wise feature representation
        if len(dims_1)>3:
            if (dims_1[-1]>1 or dims_1[-2]>1):
                embeddings1 = F.avg_pool3d(input=embeddings1, kernel_size=(1,dims_1[-2],dims_1[-1]))
            embeddings1=embeddings1.squeeze(-1).squeeze(-1)
        if len(dims_2)>3:
            if (dims_2[-1]>1 or dims_2[-2]>1):
                embeddings2 = F.avg_pool3d(input=embeddings2, kernel_size=(1,dims_2[-2],dims_1[-1]))
            embeddings2=embeddings2.squeeze(-1).squeeze(-1)


        # embeddings1: [batch x channels x frames] --> [frames x batch x channels x 1]
        emb1 = embeddings1.permute(2,0,1).unsqueeze(-1)

        # embeddings2: [batch x channels x frames] --> [frames x batch x channels x frames]
        emb2 = embeddings2.unsqueeze(0).repeat(embeddings2.size()[-1],1,1,1)

        # euclidian distance calculation
        distances = torch.abs(emb1-emb2).pow(2)

        # Softmax calculation
        softmax = torch.exp(distances)/torch.exp(torch.sum(distances,dim=-1)).unsqueeze(-1)

        # Soft nearest neighbour calculator (all frames)
        soft_nn = torch.sum(softmax*emb2,dim=-1)

        # Permute [frames x batch x channels] --> [frames x batch x channels x 1]
        soft_nn = soft_nn.unsqueeze(-1)

        # Find points of soft nn in embeddings2
        values,indices = torch.min(torch.abs(soft_nn-emb2).pow(2),dim=-1)

        indices = indices.permute(1,2,0)
        values = values.permute(1,2,0)

        # Get batch-wise T/F values
        nearest_n = embeddings2.scatter_(2,indices,1.)
        b_consistent = embeddings2 - nearest_n

        # [batch x channels x frames] --> [batch]
        b_consistent = b_consistent.sum(-1).sum(-1)

        # Non-zero elements are not consistent
        b_consistent[b_consistent==0.] = 1.
        b_consistent[b_consistent!=0.] = 0.

        return b_consistent


    '''--- Function for finding the pair-wise euclidian distance between two embeddings in batches ---'''
    '''--- The function is DEPRECATED please instead refer to `soft_nnc` '''
    def euclidian_distance(self, embeddings1, embeddings2):

        # Sum of features and transpose to match dist matrix dimensions: (batch x) #framesA x #framesB

        # norm1 shape: [batchxframesx1]
        norm1 = torch.sum(embeddings1.pow(2), dim=2)
        norm1 = norm1.unsqueeze(-1)

        # norm2 shape: [batchx1xframes]
        norm2 = torch.sum(embeddings2.pow(2), dim=2)
        norm2 = norm2.unsqueeze(1)

        # Euclidian distance in batches w/ CUDA check
        lim = torch.tensor(0.0)
        if torch.cuda.is_available():
            lim = lim.cuda()
        dist = torch.max(norm1 - 2.0 * torch.bmm(embeddings1, embeddings2.permute(0,2,1)) + norm2, lim)

        # Alignment checking
        _, indices = torch.min(dist,2)
        sorted_indices, _ = indices.sort(dim=1)
        batch_alignmed = torch.eq(indices, sorted_indices).all(1)
        # Generate query tensor
        q = torch.tensor(True)
        if torch.cuda.is_available():
            q = q.cuda()
        # Indices for batch
        batch_indices = torch.nonzero(batch_alignmed==q)

        return dist, batch_alignmed, batch_indices

    ''' --- Intersection function from Mask rcnn ---'''
    def intersect1d(self,tensor1, tensor2):
        aux = torch.cat((tensor1, tensor2),dim=0)
        aux = aux.sort()[0]
        return aux[:-1][(aux[1:] == aux[:-1]).data]

    def forward(self, x):

        # Squeeze and Recurrsion block
        tensor_shape = list(x.size())

        # Perform GlobalAvgPool operation frame-wise (for non-vector inputs)
        if (tensor_shape[3] > 1 and tensor_shape[4] > 1):
            pool = F.avg_pool3d(x,kernel_size=(1,tensor_shape[3],tensor_shape[4]),stride=1)
        else:
            pool = x

        # Reshaping to tensor of size [batch, channels, frames]
        squeezed = pool.squeeze(-1).squeeze(-1)

        # [batch, channels, frames] -> [frames, batch, channels]
        squeezed_temp = squeezed.permute(2,0,1)

        # use squeezed tensor as (2-layer) LSTM input
        lstm_out, _ = self.lstm(squeezed_temp)

        # [frames, batch, channels] -> [batch, channels, frames]
        lstm_out = lstm_out.permute(1,2,0)

        sr = lstm_out.unsqueeze(-1).unsqueeze(-1)

        # Use gates if specified
        if self.gate:
            b1_indices = self.soft_nnc(squeezed, sr)
            b2_indices = self.soft_nnc(sr, squeezed)
            if torch.nonzero(b1_indices).size()[0] != 0 and torch.nonzero(b2_indices).size()[0] != 0 :
                idx_1 = torch.nonzero(b1_indices).unsqueeze(-1)
                idx_2 = torch.nonzero(b2_indices).unsqueeze(-1)
                idx = self.intersect1d(idx_1,idx_2)
                if torch.nonzero(idx).size()[0] != 0:
                    x[idx] = x * sr.clone()
        else:
            x = x * sr.clone()

        return x
'''
===  E N D  O F  C L A S S  S Q U E E Z E _ A N D _ R E C U R S I O N ===
'''


'''
===  S T A R T  O F  C L A S S  C O N V 3 D N O T E M P O R A L ===

    [About]

        nn.Conv3d Class for creating a 3D Convolution without any temporal extend (i.e. kernel size
        is limited to a shape of 1 x k x k, where k is the kernel size).

    [Init Args]

        - in_planes: Integer for the number of channels used as input.
        - out_planes: Integer for the number of channels of the output.
        - midplanes: None. Only used to ensure continouity with class `Conv2Plus1D`.
        - stride: Integer for the kernel stride. Defaults to 1.
        - padding: Integer for zero pad. Defaults to 1.

    [Methods]

        - __init__ : Class initialiser
        - __get_downsample_stride__ : Staticmethod that returns the stride based on which the volume will be downsampled through the conv operation

'''
class Conv3DNoTemporal(nn.Conv3d):

    def __init__(self,
                 in_planes,
                 out_planes,
                 midplanes=None,
                 stride=1,
                 padding=1):

        super(Conv3DNoTemporal, self).__init__(
            in_channels=in_planes,
            out_channels=out_planes,
            kernel_size=(1, 3, 3),
            stride=(1, stride, stride),
            padding=(0, padding, padding),
            bias=False)

    @staticmethod
    def get_downsample_stride(stride):
        return (1, stride, stride)
'''
===  E N D  O F  C L A S S  C O N V 3 D N O T E M P O R A L ===
'''


'''
===  S T A R T  O F  C L A S S  B A S I C B L O C K ===

    [About]

        nn.Module Class for creating a spatio-temporal `BasicBlock` for ResNets.

    [Init Args]

        - in_planes: Integer for the number input channels to the block.
        - planes: Integer for the number of output channels to the block.
        - conv_builder: nn.Module for the Convolution type to be used.
        - stride: Integer for the kernel stride. Defaults to 1.
        - downsample: nn.Module in the case that downsampling is to be used for the residual connection.
        Defaults to None.
        - groups: Integer for the number of groups to be used.
        - base_width: Only used for contiouity with the `Bottleneck` block. Defaults to 64.
        - place: String of any [`top`,`bottom`,`skip`,`final`] specifying where the SRTG block will be placed.
        Defaults to `final`.
        - recursion: Boolean for the case that Squeeze and Recursion is used. Defaults to False.
        - gates: Boolean for the case that SR will be fused based on their cyclic consistency. Defaults to False.

    [Methods]

        - __init__ : Class initialiser
        - forward : Function for operation calling.

'''
class BasicBlock(nn.Module):

    __constants__ = ['downsample']
    expansion = 1

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, groups=1, base_width=64, place='final', recursion=False, gates=False):
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        assert (place in ['top','bottom','skip','final']), 'SAR block should be in either [`top`,`bottom`,`skip`,`final`]'

        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(
            conv_builder(inplanes, planes, midplanes, stride),
            nn.BatchNorm3d(planes),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            conv_builder(planes, planes, midplanes),
            nn.BatchNorm3d(planes)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.pos = place
        self.recursion = recursion

        if recursion:
            if place=='top':
                self.sar = Squeeze_and_Recursion(planes=inplanes, gates=gates)
            else:
                self.sar = Squeeze_and_Recursion(planes=planes, gates=gates)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        if self.pos=='top' and self.recursion:
            out = self.sar(out)
        out = self.conv2(out)
        if self.pos=='mid' and self.recursion:
            out = self.sar(out)
        if self.downsample is not None:
            identity = self.downsample(x)
            if self.pos=='skip' and self.recursion:
                identity = self.sar(identity)


        out += identity
        out = self.relu(out)
        if self.pos=='final' and self.recursion:
            out = self.sar(out)

        return out
'''
===  E N D  O F  C L A S S  B A S I C B L O C K ===
'''


'''
===  S T A R T  O F  C L A S S  B O T T L E N E C K ===

    [About]

        nn.Module Class for creating a spatio-temporal `BottleNeck` block for ResNets.

    [Init Args]

        - in_planes: Integer for the number input channels to the block.
        - planes: Integer for the number of output channels to the block.
        - conv_builder: nn.Module for the Convolution type to be used.
        - stride: Integer for the kernel stride. Defaults to 1.
        - downsample: nn.Module in the case that downsampling is to be used for the residual connection.
        Defaults to None.
        - groups: Integer for the number of groups to be used.
        - base_width: Used to define the width of the bottleneck. Defaults to 64.
        - place: String of any [`start`,`top`,`mid`,`bottom`,`skip`,`final`] specifying where the SRTG block will be placed.
        Defaults to `final`.
        - recursion: Boolean for the case that Squeeze and Recursion is used. Defaults to False.
        - gates: Boolean for the case that SR will be fused based on their cyclic consistency. Defaults to False.

    [Methods]

        - __init__ : Class initialiser
        - forward : Function for operation calling.

'''
class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, conv_builder, stride=1, downsample=None, groups=1, base_width=64, place='final', recursion=False, gates=False):

        #assert (recursion == False and gates == True), 'Temporal Gates cannot be used in recursion is not enabled'
        assert (place in ['start','top','mid','bottom','skip','final']), 'SAR block should be in either [`start`,`top`,`mid`,`bottom`,`skip`,`final`]'

        super(Bottleneck, self).__init__()
        midplanes = (inplanes * planes * 3 * 3 * 3) // (inplanes * 3 * 3 + 3 * planes)

        width = int(planes * (base_width / 64.)) * groups
        mid_width = int(midplanes * (base_width / 64.)) * groups

        # 1x1x1
        self.conv1 = nn.Sequential(
            nn.Conv3d(inplanes, width, kernel_size=1, bias=False),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=True)
        )

        # Second kernel
        self.conv2 = nn.Sequential(
            conv_builder(width, width, mid_width, stride),
            nn.BatchNorm3d(width),
            nn.ReLU(inplace=True)
        )

        # 1x1x1
        self.conv3 = nn.Sequential(
            nn.Conv3d(width, planes * self.expansion, kernel_size=1, bias=False),
            nn.BatchNorm3d(planes * self.expansion)
        )
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.pos = place
        self.recursion = recursion

        if recursion:
            if place == 'start':
                self.sar = Squeeze_and_Recursion(planes=inplanes, gates=gates)
            elif place == 'top':
                self.sar = Squeeze_and_Recursion(planes=width, gates=gates)
            if place == 'mid':
                self.sar = Squeeze_and_Recursion(planes=width, gates=gates)
            else:
                self.sar = Squeeze_and_Recursion(planes=planes * self.expansion, gates=gates)


    def forward(self, x):

        if self.pos=='start' and self.recursion:
            x = self.sar(x)

        identity = x

        out = self.conv1(x)
        if self.pos=='top' and self.recursion:
            out = self.sar(out)

        out = self.conv2(out)
        if self.pos=='mid' and self.recursion:
            out = self.sar(out)

        out = self.conv3(out)
        if self.pos=='end' and self.recursion:
            out = self.sar(out)

        if self.downsample is not None:
            identity = self.downsample(x)
            if self.pos=='res' and self.recursion:
                identity = self.sar(identity)

        out += identity
        out = self.relu(out)


        if self.pos=='final' and self.recursion:
            out = self.sar(out)

        return out
'''
===  E N D  O F  C L A S S  B O T T L E N E C K ===
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
===  S T A R T  O F  C L A S S  R 2 P L U S 1 D S T E M ===

    [About]

        nn.Sequential Class for the initial (2+1)D convolution.

    [Init Args]

        - None

    [Methods]

        - __init__ : Class initialiser
'''
class R2Plus1dStem(nn.Sequential):
    def __init__(self):
        super(R2Plus1dStem, self).__init__(
            nn.Conv3d(3, 45, kernel_size=(1, 7, 7),
                      stride=(1, 2, 2), padding=(0, 3, 3),
                      bias=False),
            nn.BatchNorm3d(45),
            nn.ReLU(inplace=True),
            nn.Conv3d(45, 64, kernel_size=(3, 1, 1),
                      stride=(1, 1, 1), padding=(1, 0, 0),
                      bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True))
'''
===  E N D  O F  C L A S S  R 2 P L U S 1 D S T E M ===
'''


'''
===  S T A R T  O F  C L A S S  V I D E O R E S N E T ===

    [About]

        nn.Module for creating the 3D ResNet.

    [Init Args]

        - block: nn.Module used as resnet building block.
        - conv_makers: List of Functions that create each layer.
        - layers: List of Integers specifying the number of blocks per layer.
        - stem: nn.Module for the Resnet stem to be used 3D/(2+1)D. Defaults to None.
        - num_classes: Integer for the dimension of the final FC layer. Defaults to 400.
        - zero_init_residual: Boolean for zero init bottleneck residual BN. Defaults to False.
        - recursion: Boolean for the case that Squeeze and Recursion is used. Defaults to False.
        - gates: Boolean for the case that SR will be fused based on their cyclic consistency.
        Defaults to False.

    [Methods]

        - __init__ : Class initialiser
        - forward: Function for performing the main sequence of operations.
        - _make_layer: Function for creating a sequence (nn.Sequential) of layers.
        - _initialise_weights: Function for weight initialisation.
'''
class VideoResNet(nn.Module):

    def __init__(self, block, conv_makers, layers,
                 stem, num_classes=400, groups=1, width_per_group=64,
                 zero_init_residual=False, recursion=False, gates=False):

        super(VideoResNet, self).__init__()
        self.inplanes = 64
        self.groups = groups
        self.base_width = width_per_group

        self.stem = stem()

        self.layer1 = self._make_layer(block, conv_makers[0], 64, layers[0], stride=1, recursion=recursion, gates=gates)

        self.layer2 = self._make_layer(block, conv_makers[1], 128, layers[1], stride=2, recursion=recursion, gates=gates)

        self.layer3 = self._make_layer(block, conv_makers[2], 256, layers[2], stride=2, recursion=recursion, gates=gates)

        self.layer4 = self._make_layer(block, conv_makers[3], 512, layers[3], stride=2, recursion=recursion, gates=gates)

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        # init weights
        self._initialise_weights()

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        # Flatten the layer to fc
        x = x.flatten(1)
        x = self.fc(x)

        return x

    def _make_layer(self, block, conv_builder, planes, blocks, stride=1, recursion=False, gates=False):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            ds_stride = conv_builder.get_downsample_stride(stride)
            downsample = nn.Sequential(
                nn.Conv3d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=ds_stride, bias=False),
                nn.BatchNorm3d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, conv_builder, stride, downsample, groups=self.groups, base_width=self.base_width, recursion=recursion, gates=gates))

        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, conv_builder, groups=self.groups, base_width=self.base_width, recursion=recursion, gates=gates))

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
'''
===  E N D  O F  C L A S S  V I D E O R E S N E T ===
'''


'''
---  S T A R T  O F  N E T W O R K  C R E A T I O N  F U N C T I O N S ---
    [About]
        All below functions deal with the creation of networks. Networks are specified based on their
        function names.
'''

def r3d_18(**kwargs):
    return VideoResNet(block=BasicBlock,conv_makers=[Conv3DSimple]*4, layers=[2, 2, 2, 2], stem=BasicStem, **kwargs)


def r3d_34(**kwargs):
    return VideoResNet(block=BasicBlock,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 6, 3], stem=BasicStem, **kwargs)


def r3d_50(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 6, 3], stem=BasicStem, **kwargs)


def r3d_101(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 23, 3], stem=BasicStem, **kwargs)


def r3d_152(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 8, 36, 3], stem=BasicStem,**kwargs)

def r3d_200(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 24, 36, 3], stem=BasicStem,**kwargs)

def r3dxt34_32d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return VideoResNet(block=BasicBlock,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 6, 3], stem=BasicStem, **kwargs)

def r3dxt50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 6, 3], stem=BasicStem, **kwargs)


def r3dxt101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 23, 3], stem=BasicStem, **kwargs)


def wide_r3d50_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 128
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 6, 3], stem=BasicStem, **kwargs)


def wide_r3d101_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 128
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 23, 3], stem=BasicStem, **kwargs)


def srtg_r3d_18(**kwargs):
    return VideoResNet(block=BasicBlock,conv_makers=[Conv3DSimple]*4, layers=[2, 2, 2, 2], stem=BasicStem, recursion=True, gates=False,  **kwargs)


def srtg_r3d_34(**kwargs):
    return VideoResNet(block=BasicBlock,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 6, 3], stem=BasicStem, recursion=True, gates=False, **kwargs)


def srtg_r3d_50(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 6, 3], stem=BasicStem, recursion=True, gates=False, **kwargs)


def srtg_r3d_101(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 23, 3], stem=BasicStem, recursion=True, gates=False, **kwargs)


def srtg_r3d_152(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 8, 36, 3], stem=BasicStem, recursion=True, gates=False, **kwargs)

def srtg_r3d_200(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 24, 36, 3], stem=BasicStem, recursion=True, gates=False, **kwargs)

def srtg_r3dxt34_32d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return VideoResNet(block=BasicBlock,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 6, 3], stem=BasicStem, recursion=True, gates=False, **kwargs)

def srtg_r3dxt50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 6, 3], stem=BasicStem, recursion=True, gates=False, **kwargs)


def srtg_r3dxt101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 23, 3], stem=BasicStem, recursion=True, gates=False, **kwargs)


def srtg_wide_r3d50_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 128
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 6, 3], stem=BasicStem, recursion=True, gates=False, **kwargs)


def srtg_wide_r3d101_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 128
    return VideoResNet(block=Bottleneck,conv_makers=[Conv3DSimple]*4, layers=[3, 4, 23, 3], stem=BasicStem, recursion=True, gates=False, **kwargs)


def r2plus1d_18(**kwargs):
    return VideoResNet(block=BasicBlock,conv_makers=[Conv2Plus1D]*4, layers=[2, 2, 2, 2], stem=R2Plus1dStem, **kwargs)


def r2plus1d_34(**kwargs):
    return VideoResNet(block=BasicBlock,conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 6, 3], stem=R2Plus1dStem, **kwargs)


def r2plus1d_50(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 6, 3], stem=R2Plus1dStem, **kwargs)


def r2plus1d_101(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 23, 3], stem=R2Plus1dStem, **kwargs)


def r2plus1d_152(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv2Plus1D]*4, layers=[3, 8, 36, 3], stem=R2Plus1dStem,**kwargs)

def r2plus1d_200(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv2Plus1D]*4, layers=[3, 24, 36, 3], stem=R2Plus1dStem,**kwargs)

def r2plus1dxt34_32d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return VideoResNet(block=BasicBlock,conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 6, 3], stem=R2Plus1dStem, **kwargs)

def r2plus1dxt50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return VideoResNet(block=Bottleneck,conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 6, 3], stem=R2Plus1dStem, **kwargs)


def r2plus1dxt101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return VideoResNet(block=Bottleneck,conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 23, 3], stem=R2Plus1dStem, **kwargs)


def wide_r2plus1d50_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 128
    return VideoResNet(block=Bottleneck,conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 6, 3], stem=R2Plus1dStem, **kwargs)


def wide_r2plus1d101_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 128
    return VideoResNet(block=Bottleneck,conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 23, 3], stem=R2Plus1dStem, **kwargs)


def srtg_r2plus1d_18(**kwargs):
    return VideoResNet(block=BasicBlock,conv_makers=[Conv2Plus1D]*4, layers=[2, 2, 2, 2], stem=R2Plus1dStem, recursion=True, gates=False,  **kwargs)


def srtg_r2plus1d_34(**kwargs):
    return VideoResNet(block=BasicBlock,conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 6, 3], stem=R2Plus1dStem, recursion=True, gates=False, **kwargs)


def srtg_r2plus1d_50(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 6, 3], stem=R2Plus1dStem, recursion=True, gates=False, **kwargs)


def srtg_r2plus1d_101(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 23, 3], stem=R2Plus1dStem, recursion=True, gates=False, **kwargs)


def srtg_r2plus1d_152(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv2Plus1D]*4, layers=[3, 8, 36, 3], stem=R2Plus1dStem, recursion=True, gates=False, **kwargs)

def srtg_r2plus1d_200(**kwargs):
    return VideoResNet(block=Bottleneck,conv_makers=[Conv2Plus1D]*4, layers=[3, 24, 36, 3], stem=R2Plus1dStem, recursion=True, gates=False, **kwargs)

def srtg_r2plus1dxt34_32d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return VideoResNet(block=BasicBlock,conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 6, 3], stem=R2Plus1dStem, recursion=True, gates=False, **kwargs)

def srtg_r2plus1dxt50_32x4d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return VideoResNet(block=Bottleneck,conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 6, 3], stem=R2Plus1dStem, recursion=True, gates=False, **kwargs)


def srtg_r2plus1dxt101_32x8d(**kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return VideoResNet(block=Bottleneck,conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 23, 3], stem=R2Plus1dStem, recursion=True, gates=False, **kwargs)


def srtg_wide_r2plus1d50_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 128
    return VideoResNet(block=Bottleneck,conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 6, 3], stem=R2Plus1dStem, recursion=True, gates=False, **kwargs)


def srtg_wide_r2plus1d101_2(pretrained=False, progress=True, **kwargs):
    kwargs['width_per_group'] = 128
    return VideoResNet(block=Bottleneck,conv_makers=[Conv2Plus1D]*4, layers=[3, 4, 23, 3], stem=R2Plus1dStem, recursion=True, gates=False, **kwargs)

'''
---  E N D  O F  N E T W O R K  C R E A T I O N  F U N C T I O N S ---
'''
