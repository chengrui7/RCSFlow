import torch
import torch.nn as nn
import torch.nn.functional as F

import math
import warnings
import copy

from torch.amp import custom_bwd, custom_fwd                        
from torch.autograd.function import Function, once_differentiable

from Models.NetUtil import *
from Lib.deform_attn_3d import deform3dattn_custom_cn
from Lib.deform_attn_2d.functions.ms_deform_attn_func import MSDeformAttnFunction

class TransFormerLayer(nn.Module):
    def __init__(self, embed_dims=256, num_levels=1, num_points=8, num_heads=4, value_dim=2,
                feedforward_channels=1024, ffn_dropout=0.0, ffn_num_fcs=2,
                act_cfg=dict(type='ReLU', inplace=True), norm_cfg=dict(type='LN')):
        super(TransFormerLayer, self).__init__()
        # init
        self.embed_dims = embed_dims
        self.num_heads = num_heads
        self.num_levels = num_levels 
        self.num_points = num_points
        self.value_dim = value_dim
        # model
        self.attention = DeformCrossAttentionCustom(embed_dims=embed_dims, num_levels=num_levels, 
                                                    num_points=num_points, value_dim=value_dim)
        self.ffn = FFN(embed_dims=embed_dims, feedforward_channels=feedforward_channels, num_fcs=ffn_num_fcs, 
                       ffn_drop=ffn_dropout,act_cfg=act_cfg)
        self.norm = build_norm_layer(norm_cfg, embed_dims)[1]

    def forward(self,
                query,
                key=None,
                value=None,
                query_pos=None,
                ref=None,
                spatial_shapes=None,
                level_start_index=None):
        """Forward function for `TransformerEncoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): The input query with shape
                [num_queries, bs, embed_dims] if
                self.batch_first is False, else
                [bs, num_queries embed_dims].
            key (Tensor): The key tensor with shape [num_keys, bs,
                embed_dims] if self.batch_first is False, else
                [bs, num_keys, embed_dims] .
            value (Tensor): The value tensor with same shape as `key`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """
        # operation_order=('attn', 'norm', 'ffn', 'norm')
        x = self.attention(query=query, key=key, value=value,
                    identity=None,
                    query_pos=query_pos,
                    key_padding_mask=None,
                    reference_points=ref,
                    spatial_shapes=spatial_shapes,
                    level_start_index=level_start_index)
        x = self.norm(x)
        x = self.ffn(x)
        x = self.norm(x)

        return x

class FFN(nn.Module):
    """FeedForward Network (MLP)"""
    def __init__(self, embed_dims=256, feedforward_channels=1024, num_fcs=2, ffn_drop=0.0,          
                 act_cfg=dict(type='ReLU', inplace=True)):
        super().__init__()
        
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        
        layers = []
        in_channels = embed_dims

        for i in range(num_fcs - 1):
            layers.append(nn.Linear(in_channels, feedforward_channels))
            # Activation Function
            layers.append(build_act_layer(self.act_cfg)[1])
            
            # Dropout
            if ffn_drop > 0:
                layers.append(nn.Dropout(ffn_drop))
            in_channels = feedforward_channels
        
        # output
        layers.append(nn.Linear(in_channels, embed_dims))
        # Dropout
        if ffn_drop > 0:
            layers.append(nn.Dropout(ffn_drop))
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.layers(x)
    
class DeformCrossAttentionCustom(nn.Module):
    """An attention module used in VoxFormer based on Deformable-Detr.

    `Deformable DETR: Deformable Transformers for End-to-End Object Detection.
    <https://arxiv.org/pdf/2010.04159.pdf>`_.

    Args:
        embed_dims (int): The embedding dimension of Attention.
            Default: 256.
        num_heads (int): Parallel attention heads. Default: 64.
        num_levels (int): The number of feature map used in
            Attention. Default: 4.
        num_points (int): The number of sampling points for
            each query in each head. Default: 4.
        im2col_step (int): The step used in image_to_column.
            Default: 64.
        dropout (float): A Dropout layer on `inp_identity`.
            Default: 0.1.
        batch_first (bool): Key, Query and Value are shape of
            (batch, n, embed_dim)
            or (n, batch, embed_dim). Default to True.
        norm_cfg (dict): Config dict for normalization layer.
            Default: None.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
        num_bev_queue (int): In this version, we only use one history BEV and one currenct BEV.
         the length of BEV queue is 2.
    """

    def __init__(self,
                 embed_dims=256,
                 num_heads=8,
                 num_levels=4,
                 num_points=4,
                 num_bev_queue=1,
                 im2col_step=64,
                 dropout=0.0,
                 value_dim=2,
                 batch_first=True):

        super().__init__()
        if embed_dims % num_heads != 0:
            raise ValueError(f'embed_dims must be divisible by num_heads, '
                             f'but got {embed_dims} and {num_heads}')
        dim_per_head = embed_dims // num_heads

        self.dropout = nn.Dropout(dropout)
        self.batch_first = batch_first
        # you'd better set dim_per_head to a power of 2
        # which is more efficient in the CUDA implementation
        def _is_power_of_2(n):
            if (not isinstance(n, int)) or (n < 0):
                raise ValueError(
                    'invalid input for _is_power_of_2: {} (type: {})'.format(
                        n, type(n)))
            return (n & (n - 1) == 0) and n != 0

        if not _is_power_of_2(dim_per_head):
            warnings.warn(
                "You'd better set embed_dims in "
                'MultiScaleDeformAttention to make '
                'the dimension of each attention head a power of 2 '
                'which is more efficient in our CUDA implementation.')

        self.im2col_step = im2col_step
        self.embed_dims = embed_dims
        self.num_levels = num_levels
        self.num_heads = num_heads
        self.num_points = num_points
        self.num_bev_queue = num_bev_queue
        self.value_dim = value_dim
        self.sampling_offsets = nn.Linear(embed_dims*self.num_bev_queue, 
                                          num_bev_queue*num_heads * num_levels * num_points * self.value_dim)
        self.attention_weights = nn.Linear(embed_dims*self.num_bev_queue,
                                           num_bev_queue*num_heads * num_levels * num_points)
        self.value_proj = nn.Linear(embed_dims, embed_dims)
        self.output_proj = nn.Linear(embed_dims, embed_dims)
        self.init_weights()

    def init_weights(self):
        # init sampling offset
        """Default initialization for Parameters of Module."""
        nn.init.constant_(self.sampling_offsets.weight, 0.)
        thetas = torch.arange(self.num_heads, dtype=torch.float32) * (2.0 * math.pi / self.num_heads)
        # check offset dim
        if self.value_dim == 2:
            grid_init = torch.stack(
                [thetas.cos(), thetas.sin()], dim=-1
            )  # (cosθ, sinθ) (num_heads, 2)
        elif self.value_dim == 3:
            grid_init = torch.stack(
                [thetas.cos(), thetas.sin(), torch.zeros_like(thetas)], dim=-1
            )  # (cosθ, sinθ, 0) (num_heads, 3)
        else:
            raise ValueError(f'Unsupported value_dim: {self.value_dim}')                   
        
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(
            self.num_heads, 1, 1, self.value_dim).repeat(1, self.num_levels*self.num_bev_queue, self.num_points, 1)
        for i in range(self.num_points):
            grid_init[:, :, i, :] *= i + 1

        self.sampling_offsets.bias.data = grid_init.view(-1)
        # init other
        nn.init.constant_(self.attention_weights.weight, 0.)
        nn.init.constant_(self.attention_weights.bias, 0.)
        
        nn.init.xavier_uniform_(self.value_proj.weight)
        nn.init.constant_(self.value_proj.bias, 0.)
        nn.init.xavier_uniform_(self.output_proj.weight)
        nn.init.constant_(self.output_proj.bias, 0.)
        self._is_init = True

    def forward(self,
                query,
                key=None,
                value=None,
                identity=None,
                query_pos=None,
                key_padding_mask=None,
                reference_points=None,
                spatial_shapes=None,
                level_start_index=None):
        """Forward Function of MultiScaleDeformAttention.

        Args:
            query (Tensor): Query of Transformer with shape
                (num_query, bs, embed_dims).
            key (Tensor): The key tensor with shape
                `(num_key, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_key, bs, embed_dims)`.
            identity (Tensor): The tensor used for addition, with the
                same shape as `query`. Default None. If None,
                `query` will be used.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`. Default
                None.
            reference_points (Tensor):  The normalized reference
                points with shape (bs, num_query, num_levels, 2),
                all elements is range in [0, 1], top-left (0,0),
                bottom-right (1, 1), including padding area.
                or (N, Length_{query}, num_levels, 4), add
                additional two dimensions is (w, h) to
                form reference boxes.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_key].
            spatial_shapes (Tensor): Spatial shape of features in
                different levels. With shape (num_levels, 2),
                last dimension represents (h, w).
            level_start_index (Tensor): The start index of each level.
                A tensor has shape ``(num_levels, )`` and can be represented
                as [0, h_0*w_0, h_0*w_0+h_1*w_1, ...].

        Returns:
             Tensor: forwarded results with shape [num_query, bs, embed_dims].
        """
        # init
        im2col_step = self.im2col_step
        if identity is None:
            identity = query
        if query_pos is not None:
            query = query + query_pos
        bsq, num_query, embed_dims = query.shape
        bsv, num_value, embed_dims = value.shape
        # print(query.shape,value.shape)
        if self.value_dim == 2:
            assert (spatial_shapes[:, 0] * spatial_shapes[:, 1]).sum() == num_value
        elif self.value_dim == 3:
            assert (spatial_shapes[:, 0] * spatial_shapes[:, 1] * spatial_shapes[:, 2]).sum() == num_value
        else:
            raise ValueError(f'Unsupported coord_dim: {self.value_dim}')
        # assert self.num_bev_queue == 2
        # query = torch.cat([value[:bs], query], -1)
        value = self.value_proj(value)
        assert key_padding_mask is None

        value = value.reshape(bsv, num_value, self.num_heads, embed_dims//self.num_heads)

        sampling_offsets = self.sampling_offsets(query)
        sampling_offsets = sampling_offsets.view(
            bsv, num_query, self.num_heads,  self.num_bev_queue, self.num_levels, self.num_points, self.value_dim)
        attention_weights = self.attention_weights(query).view(
            bsv, num_query,  self.num_heads, self.num_bev_queue, self.num_levels * self.num_points)
        attention_weights = attention_weights.softmax(-1)

        attention_weights = attention_weights.view(bsv, num_query,
                                                   self.num_heads,
                                                   self.num_bev_queue,
                                                   self.num_levels,
                                                   self.num_points)

        attention_weights = attention_weights.permute(0, 3, 1, 2, 4, 5)\
            .reshape(bsv*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points).contiguous()
        sampling_offsets = sampling_offsets.permute(0, 3, 1, 2, 4, 5, 6)\
            .reshape(bsv*self.num_bev_queue, num_query, self.num_heads, self.num_levels, self.num_points, self.value_dim)

        assert reference_points.shape[-1] == self.value_dim
        if self.value_dim == 2:
            # spatial_shapes: (num_levels, 2) -> (H, W)
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 1], spatial_shapes[..., 0]], dim=-1)  # (num_levels, 2)
        elif self.value_dim == 3:
            offset_normalizer = torch.stack(
                [spatial_shapes[..., 2], spatial_shapes[..., 1],spatial_shapes[..., 0]], -1)#hwz
        else:
            raise ValueError(f'Unsupported value_dim: {self.value_dim}')
        
        # sampling locations
        sampling_locations = reference_points[:, :, None, :, None, :] \
            + sampling_offsets \
            / offset_normalizer[None, None, None, :, None, :]
        sampling_locations = sampling_locations.contiguous()

        if self.value_dim == 2:
            output = MSDeformAttnFunction.apply(value, spatial_shapes, level_start_index, 
                        sampling_locations, attention_weights, im2col_step)
        elif self.value_dim == 3:
            output = MultiScaleDeformableAttn3DCustomFunction_fp32.apply(value, spatial_shapes, level_start_index, 
                        sampling_locations, attention_weights, im2col_step)
        else:
            raise ValueError(f'Unsupported value_dim: {self.value_dim}')
        
        # mean on bev queue (radar have no queue)
        output = output.permute(1, 2, 0)
        output = output.view(num_query, embed_dims, bsq, self.num_bev_queue)
        output = output.mean(-1)
        output = output.permute(2, 0, 1)

        output = self.output_proj(output)

        if key is not None:
            output = output * torch.sigmoid(torch.sum(key, dim=-1, keepdim=True))

        return self.dropout(output) + identity
    

class MultiScaleDeformableAttn3DCustomFunction_fp32(Function):

    @staticmethod
    @custom_fwd(cast_inputs=torch.float32, device_type='cuda')
    def forward(ctx, value, value_spatial_shapes, value_level_start_index,
                sampling_locations, attention_weights, im2col_step):
        """GPU version of multi-scale deformable attention.

        Args:
            value (Tensor): The value has shape
                (bs, num_keys, mum_heads, embed_dims//num_heads)
            value_spatial_shapes (Tensor): Spatial shape of
                each feature map, has shape (num_levels, 2),
                last dimension 2 represent (h, w)
            sampling_locations (Tensor): The location of sampling points,
                has shape
                (bs ,num_queries, num_heads, num_levels, num_points, 2),
                the last dimension 2 represent (x, y).
            attention_weights (Tensor): The weight of sampling points used
                when calculate the attention, has shape
                (bs ,num_queries, num_heads, num_levels, num_points),
            im2col_step (Tensor): The step used in image to column.

        Returns:
            Tensor: has shape (bs, num_queries, embed_dims)
        """

        ctx.im2col_step = im2col_step
        output = deform3dattn_custom_cn.ms_deform_attn_forward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            im2col_step=ctx.im2col_step)
        ctx.save_for_backward(value, value_spatial_shapes,
                              value_level_start_index, sampling_locations,
                              attention_weights)
        return output

    @staticmethod
    @once_differentiable
    @custom_bwd(device_type='cuda')
    def backward(ctx, grad_output):
        """GPU version of backward function.

        Args:
            grad_output (Tensor): Gradient
                of output tensor of forward.

        Returns:
             Tuple[Tensor]: Gradient
                of input tensors in forward.
        """
        value, value_spatial_shapes, value_level_start_index, \
            sampling_locations, attention_weights = ctx.saved_tensors
        grad_value = torch.zeros_like(value)
        grad_sampling_loc = torch.zeros_like(sampling_locations)
        grad_attn_weight = torch.zeros_like(attention_weights)

        deform3dattn_custom_cn.ms_deform_attn_backward(
            value,
            value_spatial_shapes,
            value_level_start_index,
            sampling_locations,
            attention_weights,
            grad_output.contiguous(),
            grad_value,
            grad_sampling_loc,
            grad_attn_weight,
            im2col_step=ctx.im2col_step)

        return grad_value, None, None, \
            grad_sampling_loc, grad_attn_weight, None