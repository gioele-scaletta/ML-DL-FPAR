from torch import nn

from ops.basic_ops import ConsensusModule
from ops.transforms import *
from torch.nn.init import normal_, constant_

def TSM(input, n_frames, fold_div):
    n, c, h, w = input.size()
    batch_size = n // n_frames
    input = input.view(batch_size, n_frames, c, h, w)
    fold = c // fold_div
    out = torch.zeros_like(input)
    out[:, :-1, :fold] = x[:, 1:, :fold]  # shift left
    out[:, 1:, fold: 2 * fold] = x[:, :-1, fold: 2 * fold]  # shift right
    out[:, :, 2 * fold:] = x[:, :, 2 * fold:]  # not shift

    return out
