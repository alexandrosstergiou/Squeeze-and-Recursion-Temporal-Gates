import torch
import math
import torch.nn as nn
from torch.nn.modules.utils import _triple
import torch.nn.functional as F


def soft_nnc(embeddings1,embeddings2,check=1):

    # Please keep this line for possible arithmetic underflows
    embeddings1 = embeddings1.half().double() # for improved accuracy change `half()` to `float()`
    embeddings2 = embeddings2.half().double() # but keep `double()` conversion regardless


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

    # Embeddings 1 expansion: [batch x channels x frames] --> [frames x batch x channels x 1]
    emb1 = embeddings1.permute(2,0,1).unsqueeze(-1)

    # Embeddings 2 broadcasting: [batch x channels x frames] --> [frames x batch x channels x frames]
    emb2 = embeddings2.unsqueeze(0).repeat(embeddings2.size()[-1],1,1,1)

    # Eucledian distance calculation - summed over channels #[frames x batch x frames]
    dist = torch.abs(emb2-emb1).pow(2).permute(0,1,3,2).sum(-1)
    # Minimum (frame-based) indices selection for eucledian distance
    idx_dist = dist.unsqueeze(0).argmin(-1)

    # Softmax
    exp_dist = torch.exp(dist)/torch.exp(torch.sum(dist,dim=-1)).unsqueeze(-1)
    # Minimum (frame-based) indices selection for softmax distance
    _, idx_exp = torch.min(exp_dist,dim=-1)

    # Reshaping [frames x batch] --> [batch x frames]
    idx_exp = idx_exp.permute(1,0)

    # Tensor of range(#frames): produces a sequence to be later compared with the discovered indices
    idx_e2 = torch.tensor([i for i in range(dims_1[2])]).unsqueeze(0).repeat(dims_1[0],1)

    # Find matching sequences per batch
    e1toe2 = torch.sum(idx_exp==idx_e2,dim=-1)==dims_1[2]

    # recursion for cyclic-back
    if check==2:
        return e1toe2
    else:
        e2toe1 = soft_nnc(embeddings2,embeddings1,check=2)

    # join together
    conditions = e1toe2+e2toe1

    # return only the batch indices
    return torch.where(conditions==True)[0]


if __name__ == "__main__":
    e1 = torch.rand(10,32,16)
    e2 = torch.rand(10,32,16)
    e3 = e1.clone()

    cyclic_c = soft_nnc(e1,e2)
    print('Test 1: e1 != e2 (indices):',cyclic_c.numpy(),'\n')

    cyclic_c = soft_nnc(e1,e3)
    print('Test 2: e1 == e3 (indices):',cyclic_c.numpy(),'\n')
