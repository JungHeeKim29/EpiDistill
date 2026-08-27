import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# --- Util functions ---
def masked_mean(x, mask=None, dim=None):
    if mask is None:
        return x.mean(dim=dim)
    mask = mask.float()
    masked = x * mask
    denom = mask.sum(dim=dim, keepdim=True).clamp(min=1e-6)
    return masked.sum(dim=dim, keepdim=True) / denom

def log1i(x):
    return torch.log1p(x)

def identity(x):
    return x

# --- EdgeGuidedLocalSSI ---
class EdgeGuidedLocalSSI(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = 1.0
        self.output_fn = torch.sqrt
        self.input_fn = log1i
        self.use_global = True
        self.min_samples = 6
        self.eps = 1e-5
        # Sobel filters
        delta_x = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        delta_y = torch.tensor([[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]])
        self.register_buffer('delta_x', delta_x.reshape(1, 1, 3, 3))
        self.register_buffer('delta_y', delta_y.reshape(1, 1, 3, 3))

    def get_edge(self, image, mask):
        channels = image.shape[1]
        delta_x = self.delta_x.repeat(channels, 1, 1, 1)
        delta_y = self.delta_y.repeat(channels, 1, 1, 1)
        image_Gx = F.conv2d(image, delta_x, groups=channels, padding=1) / 8
        image_Gy = F.conv2d(image, delta_y, groups=channels, padding=1) / 8
        image_Gx = image_Gx.square().mean(dim=1, keepdim=True).sqrt()
        image_Gy = image_Gy.square().mean(dim=1, keepdim=True).sqrt()
        edges = torch.sqrt(image_Gx ** 2 + image_Gy ** 2)
        edges[:, :, :3, :] = 0
        edges[:, :, -3:, :] = 0
        edges[:, :, :, :3] = 0
        edges[:, :, :, -3:] = 0
        edges[~mask.bool()] = 0
        return edges

    def forward(self, pred_depth, gt_depth, valid_mask, rgb_image, validity_mask=None):
        """
        pred_depth: (B, 1, H, W) - 모델이 예측한 depth map
        gt_depth: (B, 1, H, W) - 정답(depth ground truth)
        valid_mask: (B, H, W) - depth가 유효한 위치 마스크
        rgb_image: (B, 3, H, W) - 입력 이미지 (edge 계산용)
        validity_mask: (B, H, W) or None - 추가 유효성 마스크(옵션)
        """
        valid_mask = valid_mask.bool()
        pred_depth = self.input_fn(pred_depth.float())
        gt_depth = self.input_fn(gt_depth.float())
        B, _, H, W = pred_depth.shape
        total_errors = []
        if validity_mask is not None:
            valid_mask = valid_mask & validity_mask.bool()
        edges = self.get_edge(rgb_image, valid_mask)
        # 강한 엣지 샘플링 (간단화)
        quantile = 0.9
        reshape = 14
        edges_img = F.interpolate(edges, scale_factor=1/reshape, mode="bilinear", align_corners=False)
        edges_img_flat = edges_img.flatten(1)
        edge_thr = torch.quantile(edges_img_flat, quantile, dim=-1, keepdim=True)
        edges_mask = edges_img_flat > edge_thr
        num_samples = edges_mask.sum(dim=-1)
        min_samples = num_samples.min()
        coords = [torch.nonzero(x, as_tuple=False)[:min_samples] for x in edges_mask]
        coords = torch.stack([torch.stack([c[:,0], c[:,1]], dim=-1) for c in coords]) * reshape
        # 패치 추출 및 에러 계산 (간단화)
        kernel_size = 15
        patch_hw = kernel_size // 2
        patch_errors = []
        for b in range(B):
            for y, x in coords[b]:
                y, x = y.item(), x.item()
                y1, y2 = max(0, y-patch_hw), min(H, y+patch_hw+1)
                x1, x2 = max(0, x-patch_hw), min(W, x+patch_hw+1)
                pred_patch = pred_depth[b, :, y1:y2, x1:x2]
                gt_patch = gt_depth[b, :, y1:y2, x1:x2]
                mask_patch = valid_mask[b, y1:y2, x1:x2]
                if mask_patch.sum() >= self.min_samples:
                    err = (pred_patch - gt_patch).abs()
                    patch_errors.append(masked_mean(err, mask_patch, dim=[-2,-1]))
        if patch_errors:
            patch_errors = torch.stack(patch_errors).mean()
        else:
            patch_errors = torch.tensor(0.0, device=pred_depth.device)
        total_errors.append(self.output_fn(patch_errors.clamp(min=self.eps)))
        # global error
        if self.use_global:
            err = (pred_depth - gt_depth).abs()
            err = masked_mean(err, valid_mask, dim=[-2,-1])
            total_errors.append(self.output_fn(err.clamp(min=self.eps)))
        return torch.stack(total_errors).mean()

# --- Confidence ---
class Confidence(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = 0.1
        self.gamma = 1.0
        self.alpha = 1.0
        self.fn = F.l1_loss
        self.output_fn = torch.sqrt
        self.input_fn = identity
        self.eps = 1e-5

    def forward(self, conf_pred, depth_pred, depth_gt, valid_mask):
        """
        conf_pred: (B, N) or (B, H, W) - 예측 confidence (uncertainty 등)
        depth_pred: (B, 1, H, W) - 예측 depth
        depth_gt: (B, 1, H, W) - 정답 depth
        valid_mask: (B, H, W) - depth가 유효한 위치 마스크
        """
        B, C = depth_gt.shape[:2]
        valid_mask = valid_mask.bool()
        depth_gt = depth_gt.float().reshape(B, C, -1)
        depth_pred = depth_pred.float().reshape(B, C, -1)
        conf_pred = conf_pred.float().reshape(B, -1)
        valid_mask = valid_mask.reshape(B, -1)
        # rescale 생략 (고정)
        error = torch.abs((self.input_fn(depth_pred) - self.input_fn(depth_gt)).norm(dim=1) - conf_pred)
        losses = masked_mean(error, dim=[-1], mask=valid_mask).squeeze(dim=-1)
        losses = self.output_fn(losses)
        return losses
