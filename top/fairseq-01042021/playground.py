import torch
import numpy as np


if __name__ == "__main__":
    start_ind = 9
    x = torch.tensor([[0, 6, 10, 8, 12, 2], [0, 15, 8, 25, 6, 2]], dtype=torch.int32)
    print (x)

"""
>>> x.float()
tensor([[ 0.,  6., 10.,  8., 12.,  2.],
        [ 0., 15.,  8., 25.,  6.,  2.]])
>>> x.float().uniform_()
tensor([[0.4618, 0.8207, 0.1162, 0.5701, 0.7418, 0.0984],
        [0.4430, 0.3139, 0.5694, 0.1912, 0.0225, 0.4414]])
>>> x.float().uniform_() * torch.where(x>start_ind, 1, 0)
tensor([[0.0000, 0.0000, 0.0566, 0.0000, 0.2462, 0.0000],
        [0.0000, 0.8839, 0.0000, 0.2623, 0.0000, 0.0000]])
>>> y = x.float().uniform_() * torch.where(x>start_ind, 1, 0)
>>> y
tensor([[0.0000, 0.0000, 0.6390, 0.0000, 0.0752, 0.0000],
        [0.0000, 0.4520, 0.0000, 0.8275, 0.0000, 0.0000]])
>>> torch.argmax(y, axis=1)
tensor([2, 3])

gather everything before those indices and after (only < start_ind)
"""