# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import torch
import sys
from . import BaseWrapperDataset


class OffsetTokensDataset(BaseWrapperDataset):
    def __init__(self, dataset, offset, offset_column=None):
        super().__init__(dataset)
        self.offset = offset
        self.offset_column = offset_column

    def __getitem__(self, idx):
        if self.offset_column is not None:
            offsets = [0 for _ in range(len(self.dataset[idx]))]
            if type(self.offset_column) == int:
                assert (self.offset_column < len(offsets))
                offsets[self.offset_column] += self.offset
            elif type(self.offset_column) == list:
                for o in self.offset_column:
                    assert (o < len(offsets))
                    offsets[o] += self.offset
            else:
                sys.exit("not supported")
            return torch.add(self.dataset[idx], torch.Tensor(offsets).int())
        else:
            return self.dataset[idx] + self.offset
