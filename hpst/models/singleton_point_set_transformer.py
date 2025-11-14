from hpst.utils.options import Options
import hpst.layers.singleton_point_set_attention as psa
from typing import Tuple
from torch import Tensor, nn
import torch
    
class SegmentPointSetTransformer(nn.Module):
    def __init__(
        self,
        in_channels,
        num_classes,
        patch_embed_depth=1,
        patch_embed_channels=48,
        patch_embed_groups=6,
        patch_embed_neighbours=8,
        enc_depths=(2, 2, 6, 2),
        enc_channels=(96, 192, 384, 512),
        enc_groups=(12, 24, 48, 64),
        enc_neighbours=(16, 16, 16, 16),
        dec_depths=(1, 1, 1, 1),
        dec_channels=(48, 96, 192, 384),
        dec_groups=(6, 12, 24, 48),
        dec_neighbours=(16, 16, 16, 16),
        grid_sizes=(0.06, 0.12, 0.24, 0.48),
        attn_qkv_bias=True,
        pe_multiplier=False,
        pe_bias=True,
        attn_drop_rate=0.0,
        enable_checkpoint=False,
        unpool_backend="map",
        pos_dim=3,
        num_heads=4
    ):
        super(SegmentPointSetTransformer, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.num_stages = len(enc_depths)
        self.pos_dim = pos_dim
        assert self.num_stages == len(dec_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(dec_channels)
        assert self.num_stages == len(enc_groups)
        assert self.num_stages == len(dec_groups)
        assert self.num_stages == len(enc_neighbours)
        assert self.num_stages == len(dec_neighbours)
        assert self.num_stages == len(grid_sizes)
        self.patch_embed = psa.GVAPatchEmbed(
            in_channels=in_channels,
            embed_channels=patch_embed_channels,
            depth=patch_embed_depth,
            neighbours=patch_embed_neighbours,
            qkv_bias=attn_qkv_bias,
            attn_drop_rate=attn_drop_rate,
            enable_checkpoint=enable_checkpoint,
            num_heads=num_heads,
            pos_dim=pos_dim
        )
        enc_channels = [patch_embed_channels] + list(enc_channels)
        dec_channels = list(dec_channels) + [enc_channels[-1]]
        self.enc_stages = nn.ModuleList()
        self.dec_stages = nn.ModuleList()
        for i in range(self.num_stages):
            enc = psa.Encoder(
                depth=enc_depths[i],
                in_channels=enc_channels[i],
                embed_channels=enc_channels[i + 1],
                grid_size=grid_sizes[i],
                neighbours=enc_neighbours[i],
                qkv_bias=attn_qkv_bias,
                attn_drop_rate=attn_drop_rate,
                enable_checkpoint=enable_checkpoint,
                num_heads=num_heads,
                pos_dim=pos_dim
            )
            dec = psa.Decoder(
                depth=dec_depths[i],
                in_channels=dec_channels[i + 1],
                skip_channels=enc_channels[i],
                embed_channels=dec_channels[i],
                neighbours=dec_neighbours[i],
                qkv_bias=attn_qkv_bias,
                attn_drop_rate=attn_drop_rate,
                enable_checkpoint=enable_checkpoint,
                unpool_backend=unpool_backend,
                num_heads=num_heads,
                pos_dim=pos_dim
            )
            self.enc_stages.append(enc)
            self.dec_stages.append(dec)
        self.seg_head1 = (
            nn.Sequential(
                nn.Linear(dec_channels[0], dec_channels[0]),
                psa.PointBatchNorm(dec_channels[0]),
                nn.ReLU(inplace=True),
                nn.Linear(dec_channels[0], num_classes),
            )
            if num_classes > 0
            else nn.Identity()
        )

        self.seg_head2 = (
            nn.Sequential(
                nn.Linear(dec_channels[0], dec_channels[0]),
                psa.PointBatchNorm(dec_channels[0]),
                nn.ReLU(inplace=True),
                nn.Linear(dec_channels[0], num_classes),
            )
            if num_classes > 0
            else nn.Identity()
        )

    def forward(self, points1: Tuple[Tensor]):
        points1 = self.patch_embed(points1)
        skips = [[points1]]
        for i in range(self.num_stages):
            points1, cluster1 = self.enc_stages[i](points1)
            skips[-1].append(cluster1)  # record grid cluster of pooling
            skips.append([points1])  # record points info of current stage

        points1 = skips.pop(-1)[0]  # unpooling points info in the last enc stage
        for i in reversed(range(self.num_stages)):
            skip_points1, cluster1 = skips.pop(-1)
            points1 = self.dec_stages[i](points1, skip_points1, cluster1)
        coord1, feat1, batch1 = points1
        seg_logits1 = self.seg_head1(feat1)
        return seg_logits1
    
class PointSetTransformerInterface(nn.Module):
    def __init__(self, options: Options, feature_dim: int, output_dim: int):
        super().__init__()
        self.network = SegmentPointSetTransformer(
            feature_dim,
            output_dim,
            patch_embed_depth=options.pointnet_patch_embed_depth,
            patch_embed_channels=options.pointnet_patch_embed_channels,
            patch_embed_groups=options.pointnet_patch_embed_groups,
            patch_embed_neighbours=options.pointnet_patch_embed_neighbours,
            enc_depths=tuple(options.pointnet_enc_depths),
            enc_channels=tuple(options.pointnet_enc_channels),
            enc_groups=tuple(options.pointnet_enc_groups),
            enc_neighbours=tuple(options.pointnet_enc_neighbours),
            dec_depths=tuple(options.pointnet_dec_depths),
            dec_channels=tuple(options.pointnet_dec_channels),
            dec_groups=tuple(options.pointnet_dec_groups),
            dec_neighbours=tuple(options.pointnet_dec_neighbours),
            grid_sizes=tuple(options.pointnet_grid_sizes),
            attn_drop_rate=options.dropout,
            pos_dim=3,
            num_heads=options.pointnet_num_heads
        )

    def forward(self, p1):
        return self.network(p1)