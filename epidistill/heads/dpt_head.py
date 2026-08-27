"""Lightweight DPT head -- predicts the dense structural shift Delta d (Eq. 1).

Adapted from Ranftl et al., "Vision Transformers for Dense Prediction" (DPT).
Each level's token readout concatenates the patch tokens with the backbone CLS
token and the EpiDistill scale token; the normalised base depth D_base is fed in
as an extra channel before the final 3x3/1x1 output convolutions.

Output channels: 0 = structural shift Delta d, 1 = confidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_scratch(in_shape, out_shape):
    """1x1-free 3x3 projections bringing every level to a common width."""
    scratch = nn.Module()
    for i, in_ch in enumerate(in_shape):
        setattr(
            scratch,
            f"layer{i + 1}_rn",
            nn.Conv2d(in_ch, out_shape, kernel_size=3, stride=1, padding=1, bias=False),
        )
    return scratch


def _make_fusion_block(features, use_bn, skip_res_unit1=False):
    return FeatureFusionBlock(
        features, nn.ReLU(False), bn=use_bn, skip_res_unit1=skip_res_unit1
    )


class ResidualConvUnit(nn.Module):
    """Residual convolution module.
    """

    def __init__(self, features, activation, bn):
        """Init.

        Args:
            features (int): number of features
        """
        super().__init__()

        self.bn = bn

        self.groups=1

        self.conv1 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )
        
        self.conv2 = nn.Conv2d(
            features, features, kernel_size=3, stride=1, padding=1, bias=True, groups=self.groups
        )

        if self.bn==True:
            # GroupNorm, not BatchNorm2d: main.py's DDP setup runs
            # torch.nn.SyncBatchNorm.convert_sync_batchnorm(model) on the whole model, and
            # SyncBatchNorm's forward issues its own NCCL collective. This module is called
            # from inside a gradient-checkpointed region (model.py _decode_view_fn), which
            # re-invokes forward() during backward recompute -- an extra, rank-timing-dependent
            # SyncBatchNorm collective that races with DDP's own gradient AllReduce and hangs
            # (reproduced via faulthandler stack dump: rank stuck in
            # torch/nn/modules/_functions.py SyncBatchNorm.forward during checkpoint recompute).
            # GroupNorm normalizes within a single sample (no cross-batch/cross-rank stats), so
            # it isn't touched by convert_sync_batchnorm and is safe under checkpointing, while
            # still bounding the residual stream the same way BatchNorm did.
            self.bn1 = nn.GroupNorm(min(32, features), features)
            self.bn2 = nn.GroupNorm(min(32, features), features)

        self.activation = activation


    def forward(self, x):
        """Forward pass.

        Args:
            x (tensor): input

        Returns:
            tensor: output
        """
        
        out = self.activation(x)
        out = self.conv1(out)
        if self.bn==True:
            out = self.bn1(out)
       
        out = self.activation(out)
        out = self.conv2(out)
        if self.bn==True:
            out = self.bn2(out)

        if self.groups > 1:
            out = self.conv_merge(out)

        return out + x


class FeatureFusionBlock(nn.Module):
    """Feature fusion block.
    """

    def __init__(self, features, activation, bn=False, skip_res_unit1=False):
        super(FeatureFusionBlock, self).__init__()

        self.out_conv = nn.Conv2d(features, features, kernel_size=1, stride=1, padding=0, bias=True)

        # The deepest block receives no skip connection, so it has no first unit.
        self.skip_res_unit1 = skip_res_unit1
        if not self.skip_res_unit1:
            self.resConfUnit1 = ResidualConvUnit(features, activation, bn)
        self.resConfUnit2 = ResidualConvUnit(features, activation, bn)

    def forward(self, *xs, size=None):
        output = xs[0]
        if len(xs) == 2 and not self.skip_res_unit1:
            output = output + self.resConfUnit1(xs[1])

        output = self.resConfUnit2(output)

        modifier = {"size": size} if size is not None else {"scale_factor": 2}
        output = nn.functional.interpolate(
            output, **modifier, mode="bilinear", align_corners=True
        )
        return self.out_conv(output)
    

class DPTHead(nn.Module):
    def __init__(self, in_channels, features=256, use_bn=False,
                 out_channels=[256, 512, 1024, 1024], output_dim=2):
        super(DPTHead, self).__init__()
        
        self.projects = nn.ModuleList([
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=out_channel,
                kernel_size=1,
                stride=1,
                padding=0,
            ) for out_channel in out_channels
        ])
        
        self.resize_layers = nn.ModuleList([
            nn.ConvTranspose2d(
                in_channels=out_channels[0],
                out_channels=out_channels[0],
                kernel_size=4,
                stride=4,
                padding=0),
            nn.ConvTranspose2d(
                in_channels=out_channels[1],
                out_channels=out_channels[1],
                kernel_size=2,
                stride=2,
                padding=0),
            nn.Identity(),
            nn.Conv2d(
                in_channels=out_channels[3],
                out_channels=out_channels[3],
                kernel_size=3,
                stride=2,
                padding=1)
        ])
        
        # Each level's patch tokens are concatenated with the backbone CLS token and
        # the EpiDistill scale token before being projected back to `in_channels`.
        self.readout_projects = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(3 * in_channels, in_channels), nn.GELU())
                for _ in self.projects
            ]
        )

        self.scratch = _make_scratch(out_channels, features)

        self.scratch.refinenet1 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet2 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet3 = _make_fusion_block(features, use_bn)
        self.scratch.refinenet4 = _make_fusion_block(features, use_bn, skip_res_unit1=True)

        head_features_1 = features
        head_features_2 = 32
        
        self.scratch.output_conv1 = nn.Conv2d(head_features_1, head_features_1 // 2, kernel_size=3, stride=1, padding=1)
        
        # +1: the normalised base-depth channel D_base is concatenated before the
        # output convolutions, so the head predicts its residual, not depth from scratch.
        in_features_conv2 = head_features_1 // 2 + 1

        self.scratch.output_conv2 = nn.Sequential(
            nn.Conv2d(in_features_conv2, head_features_2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(True),
            nn.Conv2d(head_features_2, output_dim, kernel_size=1, stride=1, padding=0),
            # nn.ReLU(True),
            # nn.Identity(),
        )
        
        # Zero-init the output layer:
        #   channel 0 = structural shift  (Delta d, Eq. 1)
        #   channel 1 = confidence
        # so an untrained head reproduces the frozen backbone's structure exactly.
        with torch.no_grad():
            nn.init.constant_(self.scratch.output_conv2[-1].weight, 0)
            nn.init.constant_(self.scratch.output_conv2[-1].bias, 0)

    def forward(self, out_features, patch_h, patch_w, base_depth, scale_token):
        """out_features: list of (B, 1 + N, C), CLS first.  Returns (B, output_dim, H, W)."""
        s_tok = scale_token.unsqueeze(1) if scale_token.dim() == 2 else scale_token

        out = []
        for i, x in enumerate(out_features):
            cls_token, x_patches = x[:, 0:1, :], x[:, 1:, :]
            x = self.readout_projects[i](
                torch.cat(
                    (x_patches, cls_token.expand_as(x_patches), s_tok.expand_as(x_patches)),
                    dim=-1,
                )
            )

            x = x.permute(0, 2, 1).reshape(
                x.shape[0],
                x.shape[-1],
                patch_h,
                patch_w
            )
            # print(f"[Head] Layer {i} - shape AFTER reshape => (B,C,{patch_h},{patch_w}): {x.shape}\n")

            x = self.projects[i](x)
            x = self.resize_layers[i](x)
            out.append(x)

        layer_1, layer_2, layer_3, layer_4 = out
        layer_1_rn = self.scratch.layer1_rn(layer_1)
        layer_2_rn = self.scratch.layer2_rn(layer_2)
        layer_3_rn = self.scratch.layer3_rn(layer_3)
        layer_4_rn = self.scratch.layer4_rn(layer_4)

        path_4 = self.scratch.refinenet4(layer_4_rn, size=layer_3_rn.shape[2:])
        path_3 = self.scratch.refinenet3(path_4, layer_3_rn, size=layer_2_rn.shape[2:])
        path_2 = self.scratch.refinenet2(path_3, layer_2_rn, size=layer_1_rn.shape[2:])
        path_1 = self.scratch.refinenet1(path_2, layer_1_rn)

        out = self.scratch.output_conv1(path_1)
        # Force NCHW (contiguous) layout to avoid NHWC int32 index overflow
        # when output elements exceed INT_MAX (e.g. high-res eval with AMP)
        out = out.contiguous(memory_format=torch.contiguous_format)
        out = F.interpolate(
            out,
            (int(patch_h * 14), int(patch_w * 14)),
            mode="bilinear",
            align_corners=True
        )
        
        if base_depth.shape[-2:] != out.shape[-2:]:
            base_depth = F.interpolate(base_depth, size=out.shape[-2:], mode='nearest')
        out = torch.cat([out, base_depth], dim=1)

        return self.scratch.output_conv2(out)
