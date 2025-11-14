from typing import List, Tuple, Union
from collections import OrderedDict

import sys

import torch
from torch import Tensor, nn
from torch_geometric.nn.pool import global_mean_pool, global_add_pool
from torch_geometric.nn.unpool import knn_interpolate
from torch_scatter import scatter
from copy import deepcopy
from torch_geometric.nn.pool import voxel_grid
from torch_scatter import segment_csr, segment_coo
from torch_scatter.composite import scatter_softmax
from torch.utils.checkpoint import checkpoint
#from torch.nn.attention import flex_attention as fa
import flash_attn

# check if can be removed
from torch_cluster import knn_graph, knn

def batch2offset(batch):
    return torch.cumsum(batch.bincount(), dim=0).long()

class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, C], [B*N, L, C]
    """

    def __init__(self, embed_channels):
        super().__init__()
        self.norm = nn.BatchNorm1d(embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            return (
                self.norm(input.transpose(1, 2).contiguous())
                .transpose(1, 2)
                .contiguous()
            )
        elif input.dim() == 2:
            return self.norm(input)
        else:
            raise NotImplementedError

class GridPool(nn.Module):
    """
    Partition-based Pooling (Grid Pooling)
    """

    def __init__(self, in_channels, out_channels, grid_size, bias=False):
        super(GridPool, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.grid_size = grid_size

        self.fc = nn.Linear(in_channels, out_channels, bias=bias)
        self.norm = PointBatchNorm(out_channels)
        self.act = nn.ReLU(inplace=True)

    def forward(self, points, start=None):
        coord, feat, batch = points
        # batch = offset2batch(offset)
        feat = self.act(self.norm(self.fc(feat)))
        start = (
            segment_csr(
                coord,
                torch.cat([batch.new_zeros(1), torch.cumsum(batch.bincount(), dim=0)]),
                reduce="min",
            )
            if start is None
            else start
        )
        cluster = voxel_grid(
            pos=coord - start[batch], size=self.grid_size, batch=batch, start=0
        )
        unique, cluster, counts = torch.unique(
            cluster, sorted=True, return_inverse=True, return_counts=True
        )
        _, sorted_cluster_indices = torch.sort(cluster)
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        coord = segment_csr(coord[sorted_cluster_indices], idx_ptr, reduce="mean")
        feat = segment_csr(feat[sorted_cluster_indices], idx_ptr, reduce="max")
        batch = batch[idx_ptr[:-1]]
        # offset = batch2offset(batch)
        return [coord, feat, batch], cluster

class PointSetAttention(nn.Module):
    def __init__(
        self,
        embed_channels,
        num_heads=4,
        pos_dim=3,
        attn_drop_rate=0.0,
        qkv_bias=True,
    ):
        super(PointSetAttention, self).__init__()
        self.embed_channels = embed_channels
        self.num_heads = num_heads
        self.attn_drop_rate = attn_drop_rate
        self.qkv_bias = qkv_bias
        self.pos_dim = pos_dim

        self.qkv1 = nn.Linear(embed_channels, 2*3*embed_channels, bias=qkv_bias)
        self.qkv2 = nn.Linear(embed_channels, 2*3*embed_channels, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_rate)

        self.proj1 = nn.Linear(2*embed_channels, embed_channels)
        self.proj2 = nn.Linear(2*embed_channels, embed_channels)

        self.rpe = nn.Linear(pos_dim, embed_channels)

    def calculate_attention(self, key, query, value, batch):
        # we'll test this for now
        # TODO: add different attention methods
        #print(query.unsqueeze(0).shape)

        #window_size = 4
        #def sliding_window_causal(b, h, q_idx, kv_idx):
        #    causal_mask = q_idx >= kv_idx
        #    window_mask = q_idx - kv_idx <= window_size 
        #    return causal_mask & window_mask

        #block_mask = fa.create_block_mask(sliding_window_causal, B=None, H=None, Q_LEN=query.shape[0], KV_LEN=key.shape[0])

        #feat = fa.flex_attention(query.unsqueeze(0), key.unsqueeze(0), value.unsqueeze(0), block_mask=block_mask)

        cu_seqlens = batch2offset(batch).to(torch.int32)
        max_seqlen = torch.max(cu_seqlens[1:] - cu_seqlens[:-1]).to(torch.int32)

        feat = flash_attn.flash_attn_varlen_func(query.half(), key.half(), value.half(), 
            cu_seqlens,
            cu_seqlens,
            max_seqlen,
            max_seqlen,
            dropout_p=0.1,
            softmax_scale=1.0,
            causal=False,
            window_size=(3,3)).float()

        return feat.squeeze(0)

    def forward(self, points, view):
        coords, feats, batches = points
        feat1, feat2 = feats[view==0], feats[view==1]
        coord1, coord2 = coords[view==0], coords[view==1]

        qkv1 = self.qkv1(feat1)
        qkv1 = qkv1.reshape(-1, 2*3, self.num_heads, self.embed_channels//self.num_heads) # (N', V*3*C) => (N', V*3, H, C//H)
        qkv1 = qkv1.permute(1, 0, 2, 3) # (N'1, V*3, H, C//H) => (V*3, N'1, H, C//H)
        query11, key11, value11, query12, key12, value12 = qkv1.unbind(dim=0)

        qkv2 = self.qkv2(feat2)
        qkv2 = qkv2.reshape(-1, 2*3, self.num_heads, self.embed_channels//self.num_heads) # (N', V*3*C) => (N', V*3, H, C//H)
        qkv2 = qkv2.permute(1, 0, 2, 3) # (N'2, V*3, H, C//H) => (V*3, N'2, H, C//H)
        query21, key21, value21, query22, key22, value22 = qkv2.unbind(dim=0)

        inter_q = torch.zeros((query12.shape[0] + query21.shape[0], query12.shape[1], query12.shape[2]), device=query12.device)
        inter_k = torch.zeros((key12.shape[0] + key21.shape[0], key12.shape[1], key12.shape[2]), device=key12.device)
        inter_v = torch.zeros((value12.shape[0] + value21.shape[0], value12.shape[1], value12.shape[2]), device=value12.device)

        #print(query12.shape, query21.shape, view.sum(), (1-view).sum())

        inter_q[view==0], inter_q[view==1] = query12, query21
        inter_k[view==0], inter_k[view==1] = key12, key21
        inter_v[view==0], inter_v[view==1] = value12, value21
        
        feat11 = self.calculate_attention(key11, query11, value11, batches[view==0])
        feat22 = self.calculate_attention(key22, query22, value22, batches[view==1])

        # print("key1", key1.shape, "query1", query1.shape, "value1", value1.shape, "graph1", graph1.shape, "n1", n1, "key2", key2.shape, "query2", query2.shape, "value2", value2.shape, "graph2", graph2.shape, "n2", n2)
        inter_feat = self.calculate_attention(inter_k, inter_q, inter_v, batches)

        feat1 = torch.cat([feat11.reshape(feat11.shape[0], -1), inter_feat[view==0].reshape(inter_feat[view==0].shape[0], -1)], dim=-1)
        feat2 = torch.cat([feat22.reshape(feat22.shape[0], -1), inter_feat[view==1].reshape(inter_feat[view==1].shape[0], -1)], dim=-1)

        #print(feat1.shape)

        feat1 = self.proj1(feat1)
        feat2 = self.proj2(feat2)

        feats = torch.zeros((feat1.shape[0] + feat2.shape[0], feat1.shape[1]), device=feat1.device, dtype=feat1.dtype)
        feats[view==0], feats[view==1] = feat1, feat2

        return feats

class Block(nn.Module):
    def __init__(
        self,
        embed_channels,
        qkv_bias=True,
        attn_drop_rate=0.0,
        enable_checkpoint=False,
        pos_dim=3,
        num_heads=4
    ):
        super(Block, self).__init__()
        self.attn = PointSetAttention(
            embed_channels=embed_channels,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            pos_dim=pos_dim,
            num_heads=num_heads
        )
        self.fc11 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc12 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc31 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.fc32 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm11 = PointBatchNorm(embed_channels)
        self.norm12 = PointBatchNorm(embed_channels)
        self.norm21 = PointBatchNorm(embed_channels)
        self.norm22 = PointBatchNorm(embed_channels)
        self.norm31 = PointBatchNorm(embed_channels)
        self.norm32 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint

    def forward(self, points, view):
        coords, feats, batches = points

        identity = feats

        feats_act = torch.zeros_like(feats)

        feats_act[view==0] = self.act(self.norm11(self.fc11(feats[view==0])))
        feats_act[view==1] = self.act(self.norm12(self.fc12(feats[view==1])))

        feats = (
            self.attn((coords, feats_act, batches), view)
            if not self.enable_checkpoint
            else checkpoint(self.attn, feats, view)
        )

        feats_act = torch.zeros_like(feats)

        feats_tmp1 = self.act(self.norm21(feats[view==0]))
        feats_tmp2 = self.act(self.norm22(feats[view==1]))

        feats_act[view==0] = self.norm31(self.fc31(feats_tmp1))
        feats_act[view==1] = self.norm32(self.fc32(feats_tmp2))

        feats = identity + feats_act

        feats = self.act(feats)

        return coords, feats, batches

class BlockSequence(nn.Module):
    def __init__(
        self,
        depth,
        embed_channels,
        neighbours=16,
        qkv_bias=True,
        attn_drop_rate=0.0,
        enable_checkpoint=False,
        pos_dim=3,
        num_heads=4
    ):
        super(BlockSequence, self).__init__()

        self.neighbours = neighbours
        self.blocks = nn.ModuleList()
        for i in range(depth):
            block = Block(
                embed_channels=embed_channels,
                qkv_bias=qkv_bias,
                attn_drop_rate=attn_drop_rate,
                enable_checkpoint=enable_checkpoint,
                pos_dim=pos_dim,
                num_heads=num_heads
            )
            self.blocks.append(block)

    def forward(self, points1, points2):
        coord1, feat1, batch1 = points1
        coord2, feat2, batch2 = points2
        
        coords = torch.cat([coord1, coord2], dim=0)
        batches = torch.cat([batch1, batch2], dim=0)
        feats = torch.cat([feat1, feat2], dim=0)
        view = torch.cat([torch.zeros_like(batch1), torch.ones_like(batch2)], dim=0)

        # sort by batches, then coords. For example, batch 0 will have their
        # z coords as the sort index, then after that the second batch will 
        # go after the z coords, etc. 
        sort_w = coords[:,-1] + batches*torch.amax(coords[:,-1])
        sort_idx = torch.argsort(sort_w)

        # undo argsort - https://stackoverflow.com/questions/2483696/undo-or-reverse-argsort-python
        unsort_idx = torch.zeros_like(sort_idx)
        unsort_idx[sort_idx] = torch.arange(sort_idx.shape[0], device=sort_idx.device)

        points = (coords[sort_idx], feats[sort_idx], batches[sort_idx])
        view = view[sort_idx]

        for block in self.blocks:
            points = block(points, view)

        coords, feats, batches = points
        coords = coords[unsort_idx]
        feats = feats[unsort_idx]
        batches = batches[unsort_idx]
        view = view[unsort_idx]

        points1 = (coords[view == 0], feats[view == 0], batches[view == 0])
        points2 = (coords[view == 1], feats[view == 1], batches[view == 1])
        
        return points1, points2
    
class UnpoolWithSkip(nn.Module):
    """
    Map Unpooling with skip connection
    """

    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        bias=True,
        skip=True,
        backend="map",
    ):
        super(UnpoolWithSkip, self).__init__()
        self.in_channels = in_channels
        self.skip_channels = skip_channels
        self.out_channels = out_channels
        self.skip = skip
        self.backend = backend
        assert self.backend in ["map", "interp"]

        self.proj = nn.Sequential(
            nn.Linear(in_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )
        self.proj_skip = nn.Sequential(
            nn.Linear(skip_channels, out_channels, bias=bias),
            PointBatchNorm(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, points, skip_points, cluster=None):
        coord, feat, batch = points
        skip_coord, skip_feat, skip_batch = skip_points
        if self.backend == "map" and cluster is not None:
            feat = self.proj(feat)[cluster]
        else:
            # UNTESTED - pointops interpolation should correspond with pytorch-geometric unpool.knn_interpolate
            feat = knn_interpolate(feat, coord, skip_coord, batch, skip_batch)
        if self.skip:
            feat = feat + self.proj_skip(skip_feat)
        return [skip_coord, feat, skip_batch]


class Encoder(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        grid_size=None,
        neighbours=16,
        qkv_bias=True,
        attn_drop_rate=None,
        enable_checkpoint=False,
        pos_dim=3,
        num_heads=4
    ):
        super(Encoder, self).__init__()

        self.down1 = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.down2 = GridPool(
            in_channels=in_channels,
            out_channels=embed_channels,
            grid_size=grid_size,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
            pos_dim=pos_dim,
            num_heads=num_heads
        )

    def forward(self, points1, points2):
        points1, cluster1 = self.down1(points1)
        points2, cluster2 = self.down2(points2)
        points1, points2 = self.blocks(points1, points2)
        return points1, cluster1, points2, cluster2
    
class Decoder(nn.Module):
    def __init__(
        self,
        in_channels,
        skip_channels,
        embed_channels,
        depth,
        neighbours=16,
        qkv_bias=True,
        attn_drop_rate=None,
        enable_checkpoint=False,
        unpool_backend="map",
        num_heads=4,
        pos_dim=3
    ):
        super(Decoder, self).__init__()

        self.up1 = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend,
        )

        self.up2 = UnpoolWithSkip(
            in_channels=in_channels,
            out_channels=embed_channels,
            skip_channels=skip_channels,
            backend=unpool_backend,
        )

        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate if attn_drop_rate is not None else 0.0,
            enable_checkpoint=enable_checkpoint,
            num_heads=num_heads,
            pos_dim=pos_dim
        )

    def forward(self, points1, skip_points1, cluster1, points2, skip_points2, cluster2):
        points1 = self.up1(points1, skip_points1, cluster1)
        points2 = self.up2(points2, skip_points2, cluster2)
        return self.blocks(points1, points2)

class GVAPatchEmbed(nn.Module):
    def __init__(
        self,
        depth,
        in_channels,
        embed_channels,
        neighbours=16,
        qkv_bias=True,
        attn_drop_rate=0.0,
        enable_checkpoint=False,
        pos_dim=3,
        num_heads=4
    ):
        super(GVAPatchEmbed, self).__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels
        self.proj1 = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.proj2 = nn.Sequential(
            nn.Linear(in_channels, embed_channels, bias=False),
            PointBatchNorm(embed_channels),
            nn.ReLU(inplace=True),
        )
        self.blocks = BlockSequence(
            depth=depth,
            embed_channels=embed_channels,
            neighbours=neighbours,
            qkv_bias=qkv_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint,
            pos_dim=pos_dim,
            num_heads=num_heads
        )

    def forward(self, points1, points2):
        coord1, feat1, batch1 = points1
        coord2, feat2, batch2 = points2
        feat1 = self.proj1(feat1)
        feat2 = self.proj2(feat2)
        return self.blocks([coord1, feat1, batch1], [coord2, feat2, batch2])
