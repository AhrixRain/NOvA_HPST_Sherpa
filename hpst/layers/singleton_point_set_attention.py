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

# check if can be removed
from torch_cluster import knn_graph, knn

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

        self.qkv1 = nn.Linear(embed_channels, 3*embed_channels, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop_rate)

        self.rpe = nn.Linear(pos_dim, embed_channels)
    
    def calculate_attention_rpe(self, key, query, value, graph, coord, n):
        # print("inside", "key", key.shape, "query", query.shape, "value", value.shape, "graph", graph.shape, "n", n)
        
        coord1 = coord[graph[0],...]
        coord2 = coord[graph[1],...]
        relative_position = coord1 - coord2 # (n_neighbors*n, pos_dim)
        rpe = self.rpe(relative_position)
        rpe = rpe.reshape((-1, self.num_heads, self.embed_channels//self.num_heads)) # (n_neighbors*n, h, c)

        key = key[graph[1],...] # # (n_neighbors*n, h, c)
        query = query[graph[0],...] # (n_neighbors*n, h, c)
        value = value[graph[1],...] # (n_neighbors*n, h, c)
        # print("after", "key", key.shape, "query", query.shape, "value", value.shape, "graph", graph.shape, "n", n)
        attn = key * query + rpe # (n_neighbors*n, h, c)
        attn = attn.sum(dim=-1) # (n_neighbors * n, h)
        attn = scatter_softmax(src=attn, index=graph[0], dim=0, dim_size=n) # (n_neighbors * n, h)
        attn = attn.unsqueeze(-1)
        attn = self.attn_drop(attn)

        feat = attn * value # (n_neighbors*n, h, 1) * (n_neighbors*n, h, c) -> (n_neighbors*n, h, c)
        feat = scatter(feat, graph[0], dim=0, dim_size=n, reduce="sum") # (n_neighbors*n, h, c) -> (n, h, c)
        feat = feat.reshape(-1, feat.shape[-1]*feat.shape[-2]) # (n, h, c) -> (n, h*c) 

        return feat


    def calculate_attention_rpe_inference(self, key, query, value, graph, coord, n):
        # print("inside", "key", key.shape, "query", query.shape, "value", value.shape, "graph", graph.shape, "n", n)
        
        #key = key[graph[1],...] # # (n_neighbors*n, h, c)
        attn = key[graph[1],...]
        query = query[graph[0],...] # (n_neighbors*n, h, c)
        
        # print("after", "key", key.shape, "query", query.shape, "value", value.shape, "graph", graph.shape, "n", n)
        #attn = key * query # (n_neighbors*n, h, c)
        attn.mul_(query)

        #del key
        del query

        relative_position = coord[graph[0],...]
        coord2 = coord[graph[1],...]
        relative_position.sub_(coord2) # (n_neighbors*n, pos_dim)
        rpe = self.rpe(relative_position)
        rpe = rpe.reshape((-1, self.num_heads, self.embed_channels//self.num_heads)) # (n_neighbors*n, h, c)

        attn.add_(rpe)


        attn = attn.sum(dim=-1) # (n_neighbors * n, h)
        attn = scatter_softmax(src=attn, index=graph[0], dim=0, dim_size=n) # (n_neighbors * n, h)
        attn = attn.unsqueeze(-1)
        attn = self.attn_drop(attn)

        feat = value[graph[1],...] # (n_neighbors*n, h, c)
        feat.mul_(attn) # (n_neighbors*n, h, 1) * (n_neighbors*n, h, c) -> (n_neighbors*n, h, c)
        feat = scatter(feat, graph[0], dim=0, dim_size=n, reduce="sum") # (n_neighbors*n, h, c) -> (n, h, c)
        feat = feat.reshape(-1, feat.shape[-1]*feat.shape[-2]) # (n, h, c) -> (n, h*c) 

        return feat


    def forward(self, feat1, coord1, graph1):
        qkv1 = self.qkv1(feat1)
        qkv1 = qkv1.reshape(-1, 3, self.num_heads, self.embed_channels//self.num_heads) # (N', V*3*C) => (N', V*3, H, C//H)
        qkv1 = qkv1.permute(1, 0, 2, 3) # (N'1, V*3, H, C//H) => (V*3, N'1, H, C//H)
        query1, key1, value1 = qkv1.unbind(dim=0)

        n1 = feat1.shape[0]
        
        if self.training:
            feat1 = self.calculate_attention_rpe(key1, query1, value1, graph1, coord1, n1)
        else:
            feat1 = self.calculate_attention_rpe_inference(key1, query1, value1, graph1, coord1, n1)

        return feat1

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
        self.fc31 = nn.Linear(embed_channels, embed_channels, bias=False)
        self.norm11 = PointBatchNorm(embed_channels)
        self.norm21 = PointBatchNorm(embed_channels)
        self.norm31 = PointBatchNorm(embed_channels)
        self.act = nn.ReLU(inplace=True)
        self.enable_checkpoint = enable_checkpoint

    def forward(self, points1, graph1):
        coord1, feat1, batch1 = points1

        identity1 = feat1

        feat1 = self.act(self.norm11(self.fc11(feat1)))

        feat1 = (
            self.attn(feat1, coord1, graph1)
            if not self.enable_checkpoint
            else checkpoint(self.attn, feat1, coord1, graph1)
        )
        feat1 = self.act(self.norm21(feat1))

        feat1 = self.norm31(self.fc31(feat1))

        feat1 = identity1 + feat1

        feat1 = self.act(feat1)

        return [coord1, feat1, batch1]

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

    def forward(self, points1):
        coord1, feat1, batch1 = points1
        # reference index query of neighbourhood attention
        # for windows attention, modify reference index query method
        # reference_index, _ = pointops.knn_query(self.neighbours, coord, offset)

        graph1 = knn_graph(coord1, k=self.neighbours, batch=batch1, loop=True, flow='target_to_source')
        
        # explanation: knn_graph outputs a list of ordered pairs [[source, target]*nk]
        # where k is the number of neighbors and n is the size of x
        # This list is ordered by sources (flow option) which means that the source column
        # looks like the following: 0,0,0,0...,0 (k times) 1,1,1,1...,1 (k times)
        # hence if we reshape the target column to be of shape (-1, k) it will be the reference
        # indexes, i.e. the ith row will have the k nearest neighbors of the ith sample
        for block in self.blocks:
            points1 = block(points1, graph1)
        return points1
    
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

    def forward(self, points1):
        points1, cluster1 = self.down1(points1)
        points1 = self.blocks(points1)
        return points1, cluster1
    
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

    def forward(self, points1, skip_points1, cluster1):
        points1 = self.up1(points1, skip_points1, cluster1)
        return self.blocks(points1)

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

    def forward(self, points1):
        coord1, feat1, batch1 = points1
        feat1 = self.proj1(feat1)
        return self.blocks([coord1, feat1, batch1])