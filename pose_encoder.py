import timm
import torch
from torch import nn
from typing import Tuple, List

class PoseEncoder(nn.Module):
    def __init__(self, enc_name: str, num_inputs: int, use_coords: bool) -> None:
        super().__init__()
        self.model = timm.create_model(
            enc_name,
            in_chans=3 * num_inputs + 2 * int(use_coords),
            features_only=True,
            pretrained=True
        )
        self.use_coords = use_coords
        self.num_ch_enc = self.model.feature_info.channels()
        
    def forward(self, x: torch.Tensor, k: torch.Tensor) -> List[torch.Tensor]: 
        # Normalize input image
        x = (x - 0.45) / 0.225
        
        # Extract camera intrinsics
        fy, cy, fx, cx = k[0], k[1], k[2], k[3]
        B, _, H, W = x.shape
        
        coords = []
        if self.use_coords:
            device = x.device
            x_cords = (torch.arange(H, device=device, dtype=x.dtype)) / H
            x_cords = 1. / fx * (x_cords - cx)
            x_cords = x_cords.reshape(1, 1, -1, 1).expand(B, 1, -1, W)
            
            y_cords = torch.arange(W, device=device, dtype=x.dtype) / W
            y_cords = 1. / fy * (y_cords - cy)
            y_cords = y_cords.reshape(1, 1, 1, -1).expand(B, 1, H, -1)
            
            coords = [torch.cat([x_cords, y_cords], dim=1)]
        
        # Concatenate image and coordinates (if used) and pass through the model
        res = self.model(torch.cat([x] + coords, dim=1))
        
        return res
    
def create_pose_encoder(enc_name: str, num_inputs: int, use_coords: bool) -> PoseEncoder:
    return PoseEncoder(enc_name, num_inputs, use_coords)