# Code is based on monodepthv2, "https://github.com/nianticlabs/monodepth2"

from __future__ import absolute_import, division, print_function

import torch
import torch.nn as nn
from collections import OrderedDict
from typing import List, Optional, Tuple
from torch.nn import functional as F

class PoseDecoder(nn.Module):
    def __init__(
        self,
        num_ch_enc: List[int],
        num_input_features: int,
        num_frames_to_predict_for: Optional[int] = None,
        stride: int = 1,
        pose_tgt_ch: int = -1
    ) -> None:
        super().__init__()

        self.num_ch_enc = num_ch_enc
        self.num_input_features = num_input_features

        if num_frames_to_predict_for is None:
            num_frames_to_predict_for = num_input_features - 1
        self.num_frames_to_predict_for = num_frames_to_predict_for

        self.convs: OrderedDict[str, nn.Module] = OrderedDict()

        tgt_ch = pose_tgt_ch if pose_tgt_ch != -1 else self.num_ch_enc[(len(self.num_ch_enc) - 1) // 2]
        for i, in_ch in enumerate(self.num_ch_enc):
            self.convs[f'unify_{i}'] = nn.Sequential(
                nn.Conv2d(in_ch, tgt_ch, kernel_size=3, padding=1),
                nn.ReLU()
            )
        
        enc_ch = tgt_ch * len(self.num_ch_enc)

        self.convs["squeeze"] = nn.Conv2d(enc_ch, num_input_features * 256, kernel_size=1)
        self.convs[("pose", 0)] = nn.Conv2d(num_input_features * 256, 256, kernel_size=3, stride=stride, padding=1)
        self.convs[("pose", 1)] = nn.Conv2d(256, 256, kernel_size=3, stride=stride, padding=1)
        self.convs[("pose", 2)] = nn.Conv2d(256, 6 * num_frames_to_predict_for, kernel_size=1)

        self.relu = nn.ReLU()
        self.net = nn.ModuleList(self.convs.values())

    def downsample(self, fts: List[torch.Tensor]) -> torch.Tensor:
        tgt_sz = fts[-1].shape[2:]
        new_fts = [self.convs[f'unify_{i}'](F.adaptive_max_pool2d(f, tgt_sz)) for i, f in enumerate(fts)]
        return torch.cat(new_fts, dim=1)

    def forward(self, input_features: List[torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
        input_features = input_features[-1]
        last_features = [self.downsample(input_features)]

        cat_features = [self.relu(self.convs["squeeze"](f)) for f in last_features]
        cat_features = torch.cat(cat_features, dim=1)

        out = cat_features
        for i in range(3):
            out = self.convs[("pose", i)](out)
            if i != 2:
                out = self.relu(out)

        out = out.mean(dim=3).mean(dim=2)
        out = 0.01 * out.view(-1, self.num_frames_to_predict_for, 1, 6)

        axisangle = out[..., :3]
        translation = out[..., 3:]

        return axisangle, translation
