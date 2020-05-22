import math

import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.nn.modules.utils import _triple



class Short_Long_TCCG_Conv(nn.Module):
    r"""Applies an initial 2D purely-spatial convolution over the input volume followed by two temporal
    convolutions: one for short temporal patterns based on the produced activation volume (short) and one for
    longer patterns by downsampling and then upsampling the produced activation volume (long). The distribution of channels between the two temporal operations is user-defined. Both activation maps produced are regularised to hold representations in the same feature space throuh recurrent cells. The output is then studied in terms of the temporal cyclic consistency which if found to be present the two activations are merged. In the case that the representations are found to be too disimilar, only the short activations are returned (as the long activations tend to be more lossy given their downsampling and upsampling).

    Additional "double" mode: In addition to the above, a second mode exists to fuse the input activations with the produced ones given again their cyclic consistency in terms of their time dimension. The comparison is done after the spatio-temporal and channel dimensions are matched with that of the output through 1x1x1 convolutions and pooling (if required).

    Args:
        - in_channels (int): Number of channels in the input tensor
        - out_channels (int): Number of channels produced by the convolution
        - kernel_size (int or tuple): Size of the convolving kernel
        - stride (int or tuple, optional): Stride of the convolution. Default: 1
        - padding (int or tuple, optional): Zero-padding added to the sides of the input during their respective convolutions. Default: 0
        - groups (int): Number of groups for convolution operations. Default: 1
        - mode (string): Defines the number of TCCG in the block {'single', 'double'}. Default: 'double'
        - bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, groups=1, mode='single', bias=True):
        super(Short_Long_TCCG_Conv, self).__init__()

        # Definition for mode types ('single' / 'double')
        self.mode = mode
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        # if ints are entered, convert them to iterables, 1 -> [1, 1, 1]
        kernel_size = _triple(kernel_size)
        stride = _triple(stride)
        padding = _triple(padding)

        # decomposing the parameters into spatial and temporal components by
        # masking out the values with the defaults on the axis that
        # won't be convolved over. This is necessary to avoid unintentional
        # behavior such as padding being added twice
        spatial_kernel_size =  [1, kernel_size[1], kernel_size[2]]
        spatial_stride =  [1, stride[1], stride[2]]
        spatial_padding =  [0, padding[1], padding[2]]

        temporal_kernel_size = [kernel_size[0], 1, 1]
        temporal_stride =  [stride[0], 1, 1]
        temporal_padding =  [padding[0], 0, 0]

        # compute the number of intermediary channels (M) using formula
        intermed_channels = int(math.floor((kernel_size[0] * kernel_size[1] * kernel_size[2] * in_channels * out_channels)/(2 * (kernel_size[0] * in_channels + kernel_size[1] * kernel_size[2] * out_channels))))

        intermed_channels = math.floor(intermed_channels/groups) * groups

        self.bn1 = nn.BatchNorm3d(intermed_channels)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU()

        #out_channels_short = int(out_channels*r)

        #out_channels_long = int(out_channels - out_channels_short)

        # Definition for the Downsampling and Upsampling operations
        self.pool = nn.AvgPool3d(kernel_size=(2,2,2))
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # Channel codition for studying spatio-temporal cyclic consistency in double mode
        # Create 1x1x1 Concolution layer if necessary
        if (self.mode == 'double'):
            if (in_channels != out_channels):
                self.c_conv = nn.Conv3D(in_channels, out_channels, (1,1,1))
            else:
                self.c_conv = None

        # Short sequence temporal convolution.
        self.temporal_conv_short = nn.Conv3d(in_channels, intermed_channels, temporal_kernel_size, stride=temporal_stride, padding=temporal_padding, groups=groups, bias=bias)

        # Long sequence temporal convolution.
        self.temporal_conv_long = nn.Conv3d(in_channels, intermed_channels, temporal_kernel_size,stride=temporal_stride, padding=temporal_padding, groups=groups, bias=bias)

        # Spatial convolution
        self.spatial_conv = nn.Conv3d(intermed_channels, out_channels, spatial_kernel_size,stride=spatial_stride, padding=spatial_padding, groups=groups, bias=bias)




    def forward(self, x):

        x_short = self.relu(self.bn1(self.temporal_conv_short(x)))
        x_long = self.relu(self.bn1(self.upsample(self.temporal_conv_long(self.pool(x)))))

        x_spatial_short = self.spatial_conv(x_short)

        x_spatial_long = self.spatial_conv(x_long)


        x_out = x_spatial_short

        # Check cyclic consistency
        _, _, b_indices = self.euclidian_distance(x_spatial_long, x_spatial_short)
        #if len(b_indices.size()) != 0 :
        #    x_out[b_indices] += x_spatial_long[b_indices]


        # Case of double TCCG gates
        if (self.mode == 'double'):

            # ensuring channel size matching
            if (self.c_conv):
                x = self.c_conv(x)

            _,_,b_indices = self.euclidian_distance(x,x_out)
            if len(b_indices.size()) != 0 :
                x_out[b_indices] += x[b_indices]


        x_out = self.bn2(x_out)
        x_out = self.relu(x_out)

        return x_out



    def soft_nnc(self,embeddings1,embeddings2):

        # Assume inputs shapes of (batch x) x channels x frames x height x width
        _, _, f1, h1, w1 = embeddings1.shape
        _, _, f2, h2, w2 = embeddings2.shape

        # Pooling Height and width to create frame-wise feature representation
        embeddings1 = F.avg_pool3d(input=embeddings1, kernel_size=(1,h1,w1)).squeeze(-1).squeeze(-1)
        embeddings2 = F.avg_pool3d(input=embeddings2, kernel_size=(1,h2,w2)).squeeze(-1).squeeze(-1)

        # embeddings1: [batch x channels x frames] --> [frames x batch x channels x 1]
        emb1 = embeddings1.permute(2,0,1).unsqueeze(-1)

        # embeddings2: [batch x channels x frames] --> [frames x batch x channels x frames]
        emb2 = embeddings2.unsqueeze(0).repeat(embeddings2.size()[-1],1,1,1)

        # euclidian distance calculation
        distances = torch.abs(emb1-emb2).pow(2)
        distances = distance.permute(1,2,0,3)

        # Softmax calculation
        import sys
        softmax_numerator = sys.float_info.epsilon**(-(distances))
        softmax_denominator = sys.float_info.epsilon**(-torch.sum(distances,dim=-1))
        softmax = softmax_numerator / softmax_denominator.unsqueeze(-1)

        # Soft nearest neighbour calculator (all frames)
        emb2 = emb2.permute(1,2,0,3)
        soft_nn = torch.sum(softmax*emb2,dim=-1)

        # euclidian distances for closesest soft NN points in emb2
        soft_nn = soft_nn.unsqueeze(-1)
        dist = torch.abs(soft_nn-emb2).pow(2)
        values,indices = torch.min(dist,-1)

        # Get batch-wise T/F values
        nearest_n = embeddings2[indices]
        b_consistent = embeddings2 - nearest_n

        # [batch x channels x frames] --> [batch]
        b_consistent = b_consistent.sum(-1).sum(-1)

        # Non-zero elements are not consistent
        b_consistent[b_consistent==0.] = 1.
        b_consistent[b_consistent!=0.] = 0.

        # Return batches that there is temporal consistency
        return b_consistent



    '''Function for finding the pair-wise euclidian distance between two embeddings in batches'''
    def euclidian_distance(self, embeddings1, embeddings2):

        # Assume inputs shapes of (batch x) x channels x frames x height x width
        _, _, f1, h1, w1 = embeddings1.shape
        _, _, f2, h2, w2 = embeddings2.shape

        # Pooling Height and width to create frame-wise feature representation
        embeddings1 = F.avg_pool3d(input=embeddings1, kernel_size=(1,h1,w1)).squeeze(-1).squeeze(-1)
        embeddings2 = F.avg_pool3d(input=embeddings2, kernel_size=(1,h2,w2)).squeeze(-1).squeeze(-1)


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
