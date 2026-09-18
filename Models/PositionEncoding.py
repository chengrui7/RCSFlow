import torch
import torch.nn as nn
import torch.nn.functional as F
    
class LearnedPositionEncoding3D(nn.Module):
    def __init__(self, num_feats, depth_num_embed=50, height_num_embed=50, width_num_embed=50):
        super().__init__()

        self.depth_embed = nn.Embedding(depth_num_embed, num_feats)  
        self.height_embed = nn.Embedding(height_num_embed, num_feats) 
        self.width_embed = nn.Embedding(width_num_embed, num_feats)    
        # init
        self.num_feats = num_feats
        self.depth_num_embed = depth_num_embed
        self.height_num_embed = height_num_embed
        self.width_num_embed = width_num_embed
        
    def forward(self, batch_size=None, mask=None):
        """
        Args:
            mask: [batch_size, height, width, depth] 
                ByteTensor mask. Non-zero values representing
                ignored positions, while zero values means valid positions
                for this image.
        Returns:
            pos:  [batch_size, num_feats*3, depth, height, width]
                Returned position embedding with shape
                [bs, num_feats*3, h, w, d].
        """
        if mask is not None:
            bs = mask.shape[0]
            depth, height, width = mask.shape[-3:]
            device = mask.device
        else:
            assert batch_size is not None, \
                "batch_size must be provided when mask is None"
            bs = batch_size
            depth = self.depth_num_embed
            height = self.height_num_embed
            width = self.width_num_embed
            device = self.depth_embed.weight.device
        x_coords = torch.arange(depth, device=device)  # front
        y_coords = torch.arange(height, device=device)   # right
        z_coords = torch.arange(width, device=device)   # up
        
        # embed
        x_embed = self.depth_embed(x_coords)    # [width, num_feats]
        y_embed = self.height_embed(y_coords)   # [height, num_feats]
        z_embed = self.width_embed(z_coords)    # [depth, num_feats]

        # x_embed: [1, 1, width, num_feats] -> [depth, height, width, num_feats]
        x_embed = x_embed.view(depth, 1, 1, -1).expand(depth, height, width, -1)
        # y_embed: [1, height, 1, num_feats] -> [depth, height, width, num_feats]
        y_embed = y_embed.view(1, height, 1, -1).expand(depth, height, width, -1)
        # z_embed: [depth, 1, 1, num_feats] -> [depth, height, width, num_feats]
        z_embed = z_embed.view(1, 1, width, -1).expand(depth, height, width, -1)
        pos = torch.cat([x_embed, y_embed, z_embed], dim=-1)  # [depth, height, width, num_feats*3]
        
        pos = pos.view(depth*height*width, -1).contiguous()  
        pos = pos.unsqueeze(0).expand(bs, -1, -1)  # [bs, depth*height*width, c]
        
        return pos
    
    def __repr__(self):
        # print
        return (f"{self.__class__.__name__}("
                f"num_feats={self.num_feats!r}, "
                f"height_num_embed={self.height_num_embed!r}, "
                f"width_num_embed={self.width_num_embed!r}, "
                f"depth_num_embed={self.depth_num_embed!r})")