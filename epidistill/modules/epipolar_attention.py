"""Depth-Guided Epipolar Attention (EpiDistill, Sec. 4.1).

A reference-view query attends only to points sampled along its epipolar line in
the source view (Eq. 2).  During multi-view training the ground-truth depth gives
the exact correspondence p*_src (Eq. 3); a spatial Gaussian bias B_ij (Eq. 4)
concentrates a *target* attention map A* around it, and the cross-entropy in
Eq. 5 (``L_attn``) distils that target into the model's own attention.

``rectified=True`` is the single-view / Rectified-Stereo-Token regime of Sec. 4.2:
R = I and t = [b, 0, 0], so the epipolar line collapses to the query's own
horizontal scanline (y = y0) and the sampling grid is baseline-agnostic.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

def skew_symmetric(v):
    """
    Convert vector to skew symmetric matrix
    v: (B, 3)
    Returns: (B, 3, 3)
    """
    B = v.shape[0]
    zero = torch.zeros(B, device=v.device, dtype=v.dtype)
    
    #     [ 0, -z,  y]
    #     [-y,  x,  0]
    
    x, y, z = v[:, 0], v[:, 1], v[:, 2]
    
    S = torch.stack([
        zero, -z, y,
        z, zero, -x,
        -y, x, zero
    ], dim=1).reshape(B, 3, 3)
    
    return S

def compute_fundamental_matrix(K_ref, K_src, T_ref2src):
    """
    Compute Fundamental Matrix F from Ref to Source.
    x_src^T F x_ref = 0
    """
    # Guard: Check for singular K (near-zero fx or fy on diagonal)
    K_ref_diag = K_ref.diagonal(dim1=-2, dim2=-1)  # (B, 3)
    K_src_diag = K_src.diagonal(dim1=-2, dim2=-1)  # (B, 3)
    
    ref_singular = (K_ref_diag[:, :2].abs() < 1e-6).any()
    src_singular = (K_src_diag[:, :2].abs() < 1e-6).any()
    
    if ref_singular or src_singular:
        if ref_singular:
            print(f"⚠️ [compute_fundamental_matrix] K_ref has near-zero diagonal (singular)!")
            print(f"   K_ref diag: {K_ref_diag}")
            print(f"   K_ref full:\n{K_ref}")
        if src_singular:
            print(f"⚠️ [compute_fundamental_matrix] K_src has near-zero diagonal (singular)!")
            print(f"   K_src diag: {K_src_diag}")
            print(f"   K_src full:\n{K_src}")
        return torch.eye(3, device=K_ref.device, dtype=K_ref.dtype).unsqueeze(0).expand(K_ref.shape[0], -1, -1)
    
    R = T_ref2src[:, :3, :3]
    t = T_ref2src[:, :3, 3] # (B, 3)
    
    t_skew = skew_symmetric(t)
    E = t_skew @ R
    
    K_ref_inv = torch.inverse(K_ref)
    K_src_inv = torch.inverse(K_src)
    
    F = K_src_inv.transpose(1, 2) @ E @ K_ref_inv
    
    return F

def get_epipolar_lines_cross(K_ref, K_src, T_ref2src, p_ref):
    """
    Compute epipolar lines in Source Image for points p_ref in Ref Image.
    l = F p_ref (l: [a, b, c] -> ax + by + c = 0)
    """
    F = compute_fundamental_matrix(K_ref, K_src, T_ref2src) # (B, 3, 3)
    lines = F @ p_ref # (B, 3, N)
    
    ab_norm = torch.norm(lines[:, :2, :], dim=1, keepdim=True) + 1e-5
    lines = lines / ab_norm
    
    return lines


def get_line_image_intersection(lines, H, W):
    """
    Compute intersection of epipolar lines with image bounds.
    Returns start/end points of the segment within the image.
    """
    B, _, N = lines.shape
    
    a = lines[:, 0, :]
    b = lines[:, 1, :]
    c = lines[:, 2, :]
    
    eps = 1e-5
    sign_a = torch.sign(a)
    sign_a[sign_a == 0] = 1.0
    a_safe = torch.where(a.abs() < eps, sign_a * eps, a)
    
    sign_b = torch.sign(b)
    sign_b[sign_b == 0] = 1.0
    b_safe = torch.where(b.abs() < eps, sign_b * eps, b)
    
    y_x0 = -c / b_safe
    x1, y1 = torch.zeros_like(y_x0), y_x0
    
    y_xW = (-c - a * (W - 1)) / b_safe
    x2, y2 = torch.full_like(y_xW, W - 1), y_xW
    
    x_y0 = -c / a_safe
    x3, y3 = x_y0, torch.zeros_like(x_y0)
    
    x_yH = (-c - b * (H - 1)) / a_safe
    x4, y4 = x_yH, torch.full_like(x_yH, H - 1)
    
    pts = torch.stack([
        torch.stack([x1, y1], dim=-1),
        torch.stack([x2, y2], dim=-1),
        torch.stack([x3, y3], dim=-1),
        torch.stack([x4, y4], dim=-1)
    ], dim=2) # (B, N, 4, 2)
    
    pad = 2.0
    in_bound = (pts[..., 0] >= -pad) & (pts[..., 0] <= W - 1 + pad) & \
               (pts[..., 1] >= -pad) & (pts[..., 1] <= H - 1 + pad)
               
    dx = -b
    dy = a
    proj = pts[..., 0] * dx.unsqueeze(2) + pts[..., 1] * dy.unsqueeze(2)
    
    proj_min = proj.clone()
    proj_min[~in_bound] = float('inf')
    
    proj_max = proj.clone()
    proj_max[~in_bound] = float('-inf')
    
    idx_min = torch.argmin(proj_min, dim=-1)
    idx_max = torch.argmax(proj_max, dim=-1)
    
    valid_line = (proj_min.min(dim=-1)[0] < float('inf')) & (proj_max.max(dim=-1)[0] > float('-inf'))
    
    p_start = torch.gather(pts, 2, idx_min.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)).squeeze(2)
    p_end = torch.gather(pts, 2, idx_max.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, 2)).squeeze(2)
    
    return p_start, p_end, valid_line, dx, dy

def sample_lines_uniform(lines, H, W, num_samples):
    B, _, N = lines.shape
    device = lines.device
    
    p_start, p_end, valid_line, _, _ = get_line_image_intersection(lines, H, W)
    
    p_start[~valid_line] = 0.0
    p_end[~valid_line] = 0.0
    
    t = torch.linspace(0, 1, num_samples, device=device).view(1, 1, num_samples, 1)
    points = p_start.unsqueeze(2) * (1 - t) + p_end.unsqueeze(2) * t
    
    points = torch.nan_to_num(points, nan=0.0, posinf=1e4, neginf=-1e4)
    
    line_len = torch.norm(p_end - p_start, dim=-1)
    valid_line = valid_line & (line_len > 1.0)
    valid_samples = valid_line.unsqueeze(-1).expand(-1, -1, num_samples)
    
    return points, valid_samples

def get_epipolar_sampling_grid(
    query_coords,
    K_ref,
    K_src,
    T_ref2src,
    img_shape,          # (H_s, W_s)
    num_samples=32,
):
    """Sampling grid (B, H, W, S, 2) for the epipolar attention, in [-1, 1].

    ``num_samples`` points are spread uniformly along the visible segment of each
    query's epipolar line (Eq. 2).  The depth guidance of Eq. 3-4 does NOT narrow
    this sampling range -- it enters as the spatial bias B_ij on the attention
    logits, so the model still sees the whole line and only its *attention* is
    concentrated.
    """
    if query_coords.dim() == 4:
        B, H, W, _ = query_coords.shape
        N = H * W
    else:
        B, N, _ = query_coords.shape
        H = img_shape[0] if img_shape else int(math.sqrt(N))
        W = img_shape[1] if img_shape else int(math.sqrt(N))
    device = query_coords.device

    p_ref_pix = query_coords.view(B, H * W, 2).permute(0, 2, 1)          # (B, 2, N)
    ones = torch.ones(B, 1, H * W, device=device)
    p_ref = torch.cat([p_ref_pix, ones], dim=1)                          # (B, 3, N)

    lines = get_epipolar_lines_cross(K_ref, K_src, T_ref2src, p_ref)     # (B, 3, N)
    grid, valid_samples = sample_lines_uniform(lines, img_shape[0], img_shape[1], num_samples)

    grid = grid.view(B, H, W, num_samples, 2)
    
    H_s, W_s = img_shape
    grid_norm = torch.zeros_like(grid)
    grid_norm[..., 0] = 2.0 * grid[..., 0] / max(W_s - 1, 1) - 1.0
    grid_norm[..., 1] = 2.0 * grid[..., 1] / max(H_s - 1, 1) - 1.0
    
    return grid_norm, valid_samples

class DepthGuidedEpipolarAttention(nn.Module):
    """
    Epipolar Attention Block.
    Constrains attention to epipolar lines defined by relative pose.
    """
    def __init__(self, dim, num_heads=8, num_samples=32, grad_checkpointing=True, rope=None):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.num_samples = num_samples
        self.scale = (dim // num_heads) ** -0.5
        self.grad_checkpointing = grad_checkpointing
        self.rope = rope # Store rope module

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.o_proj = nn.Linear(dim, dim)
        

    def forward(
        self,
        query_feat, # (B, C, H, W)
        key_feat,   # (B, C, H_s, W_s)
        value_feat, # (B, C, H_s, W_s)
        K_ref,      # (B, 3, 3)
        K_src,      # (B, 3, 3)
        T_ref2src,  # (B, 4, 4)
        depth=None,
        rectified=False, # New argument
        pos=None, # Position Encoding
        apply_depth_bias=False, # New argument: Bias attention towards projected depth
        return_matching_loss=False, # Return the cross-entropy attention loss (Eq. 5)
    ):
        if self.grad_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                query_feat, key_feat, value_feat,
                K_ref, K_src, T_ref2src, depth, rectified, pos, apply_depth_bias, return_matching_loss,
                use_reentrant=False
            )
        else:
            return self._forward_impl(query_feat, key_feat, value_feat, K_ref, K_src, T_ref2src, depth, rectified, pos, apply_depth_bias, return_matching_loss)

    def _forward_impl(self, query_feat, key_feat, value_feat, K_ref, K_src, T_ref2src, depth, rectified, pos, apply_depth_bias, return_matching_loss):
        B, C, H, W = query_feat.shape
        H_s, W_s = key_feat.shape[-2:]
        
        if rectified:
            # Sample uniformly along the same row (y)
            
            
            # query y ranges [0, H-1]. Normalized: 2*y/(H-1) - 1.
            
            y_norm = torch.linspace(-1, 1, steps=H, device=query_feat.device).view(1, H, 1, 1).expand(B, H, W, self.num_samples)
            
            x_norm = torch.linspace(-1, 1, steps=self.num_samples, device=query_feat.device).view(1, 1, 1, self.num_samples).expand(B, H, W, self.num_samples)
            
            sampling_grid = torch.stack([x_norm, y_norm], dim=-1) # (B, H, W, S, 2)
            
            # All valid for rectified logic
            valid_mask = torch.ones(B, H, W, self.num_samples, dtype=torch.bool, device=sampling_grid.device)
            
            
        else:
            y, x = torch.meshgrid(torch.arange(H, device=query_feat.device), torch.arange(W, device=query_feat.device), indexing='ij')
            query_coords = torch.stack([x, y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).float() # (B, H, W, 2)
            
            sampling_grid, valid_mask = get_epipolar_sampling_grid(
                query_coords,
                K_ref, K_src, T_ref2src,
                img_shape=(H_s, W_s),
                num_samples=self.num_samples,
            ) # (B, H, W, S, 2)
        
        if torch.isnan(sampling_grid).any():
             sampling_grid = torch.nan_to_num(sampling_grid)

        N = H * W
        flat_grid = sampling_grid.view(B, N, self.num_samples, 2)
        
        # Clamp grid for stability (grid_sample can handle outside, but gradients might be huge)
        flat_grid = flat_grid.clamp(-2.0, 2.0)

        sampled_k = F.grid_sample(key_feat, flat_grid, align_corners=True, mode='bilinear', padding_mode='zeros')
        sampled_v = F.grid_sample(value_feat, flat_grid, align_corners=True, mode='bilinear', padding_mode='zeros')
        
        q = self.q_proj(query_feat.permute(0, 2, 3, 1)).view(B, N, C)
        
        k = self.k_proj(sampled_k.permute(0, 2, 3, 1)) 
        v = self.v_proj(sampled_v.permute(0, 2, 3, 1))
        
        head_dim = self.dim // self.num_heads
        
        q = q.view(B, N, self.num_heads, head_dim).transpose(1, 2) # (B, nH, N, d)
        
        k_reshaped = k.view(B, N, self.num_samples, self.num_heads, head_dim).permute(0, 3, 1, 2, 4) # (B, nH, N, S, d)
        v = v.view(B, N, self.num_samples, self.num_heads, head_dim).permute(0, 3, 1, 2, 4) # (B, nH, N, S, d)
        
        # --- Apply RoPE for Geometric Bias ---
        if self.rope is not None and pos is not None:
            q = self.rope(q, pos)
            
            if rectified:
                # Query pos is (B, N, 2) -> (B, H*W, 2) -> (y, x)
                # Key positions: y is same as Query y. x is 0..S-1.
                
                q_y = pos[..., 0].long() # (B, N)
                
                s_idx = torch.arange(self.num_samples, device=q.device).unsqueeze(0).unsqueeze(0).expand(B, N, -1) # (B, N, S)
                
                k_y = q_y.unsqueeze(-1).expand(-1, -1, self.num_samples) # (B, N, S)
                
                pos_k = torch.stack([k_y, s_idx], dim=-1).view(B, N*self.num_samples, 2)
                
                k_flat = k_reshaped.permute(0, 1, 2, 4, 3).flatten(2, 3) # (B, nH, N*S, d) 
                # Note: k_reshaped was (B, nH, N, S, d). Flattening N, S maps to N*S
                
                k_rope = self.rope(k_flat, pos_k)
                
                # Then permute to (B, nH, N, d, S) for attention
                k = k_rope.view(B, self.num_heads, N, self.num_samples, head_dim).permute(0, 1, 2, 4, 3) # (B, nH, N, d, S)
                
            else:
                
                
                grid_norm = flat_grid # (B, N, S, 2)
                
                x_norm = grid_norm[..., 0]
                y_norm = grid_norm[..., 1]
                
                # Use W_s, H_s from key_feat
                x_pix = (x_norm + 1.0) * (W_s - 1.0) / 2.0
                y_pix = (y_norm + 1.0) * (H_s - 1.0) / 2.0
                
                x_idx = x_pix.round().long().clamp(0, W_s - 1)
                y_idx = y_pix.round().long().clamp(0, H_s - 1)
                
                # k_reshaped: (B, nH, N, d, S) -> permute to (B, nH, N, S, d) -> flatten N, S
                
                pos_k = torch.stack([y_idx, x_idx], dim=-1).view(B, N*self.num_samples, 2)
                
                k_flat = k_reshaped.permute(0, 1, 2, 4, 3).flatten(2, 3)
                
                k_rope = self.rope(k_flat, pos_k)
                
                k = k_rope.view(B, self.num_heads, N, self.num_samples, head_dim).permute(0, 1, 2, 4, 3)

        else:
             k = k_reshaped.permute(0, 1, 2, 4, 3)
        
        q = q.unsqueeze(3)
        
        attn = (q @ k) * self.scale

        matching_loss = None
        # --- GT Depth Bias Injection ---
        # Guard: skip depth bias entirely if K_ref is singular (near-zero fx/fy)
        _k_ref_ok = (K_ref.diagonal(dim1=-2, dim2=-1)[:, :2].abs() > 1e-6).all()
        if apply_depth_bias and depth is not None and not rectified and _k_ref_ok:
             
             if depth.dim() == 3: depth_flat = depth.view(B, -1)
             elif depth.dim() == 4: depth_flat = depth.view(B, -1)
             else: depth_flat = depth
             
             # Reuse helper logic or reimplement for efficiency
             # p_ref_pix was generated in get_epipolar_sampling_grid but not returned.
             # Regenerate p_ref_pix
             y_ref, x_ref = torch.meshgrid(torch.arange(H, device=query_feat.device), torch.arange(W, device=query_feat.device), indexing='ij')
             p_ref_pix = torch.stack([x_ref, y_ref], dim=-1).view(1, N, 2).expand(B, N, 2).transpose(1, 2) # (B, 2, N)
             
             ones = torch.ones(B, 1, N, device=query_feat.device)
             p_ref_homo = torch.cat([p_ref_pix, ones], dim=1) # (B, 3, N)
             
             K_ref_inv = torch.inverse(K_ref)
             Ray_ref = K_ref_inv @ p_ref_homo 
             R = T_ref2src[:, :3, :3]
             t = T_ref2src[:, :3, 3:4]
             
             P_src = (R @ Ray_ref) * depth_flat.unsqueeze(1) + t # (B, 3, N)
             p_src_homo = K_src @ P_src
             
             z_src = p_src_homo[:, 2:3, :]
             valid_proj = z_src > 1e-3
             
             p_src_pix = p_src_homo[:, :2, :] / (z_src + 1e-6) # (B, 2, N)
             
             
             p_src_norm = torch.zeros_like(p_src_pix)
             p_src_norm[:, 0, :] = 2.0 * p_src_pix[:, 0, :] / max(W_s - 1, 1) - 1.0
             p_src_norm[:, 1, :] = 2.0 * p_src_pix[:, 1, :] / max(H_s - 1, 1) - 1.0
             
             target_points = p_src_norm.transpose(1, 2).unsqueeze(2) # (B, N, 1, 2)
             
             dist_sq = ((flat_grid - target_points)**2).sum(dim=-1)
             
             
             bias_scale = 50.0
             attn_bias = -bias_scale * dist_sq # (B, N, S)
             
             valid_proj_flat = valid_proj.squeeze(1).unsqueeze(-1) # (B, N, 1)
             
             attn_bias = torch.where(valid_proj_flat, attn_bias, torch.zeros_like(attn_bias))

             if return_matching_loss:
                 
                 valid_mask_flat = valid_mask.view(B, N, self.num_samples) # (B, N, S)
                 gt_target_idx = torch.argmax(attn_bias, dim=-1) # (B, N)
                 
                 attn_logits = attn.mean(dim=1).squeeze(2) # (B, N, S)
                 
                 ce_loss = F.cross_entropy(
                     attn_logits.reshape(-1, self.num_samples), 
                     gt_target_idx.reshape(-1), 
                     reduction='none'
                 ).reshape(B, N)
                 
                 # A line is valid if it has at least one valid sample, but we also rely on valid_proj
                 any_valid = valid_mask_flat.any(dim=-1) # (B, N)
                 valid_matching = valid_proj_flat.squeeze(-1) & any_valid # (B, N)
                 
                 if valid_matching.sum() > 0:
                     matching_loss = (ce_loss * valid_matching.float()).sum() / valid_matching.float().sum()
                 else:
                     matching_loss = torch.tensor(0.0, device=query_feat.device)

             attn = attn + attn_bias.unsqueeze(1).unsqueeze(3)
        

        
        
        valid_mask_flat = valid_mask.view(B, N, self.num_samples) # (B, N, S)
        valid_mask_expanded = valid_mask_flat.unsqueeze(1).unsqueeze(3).expand(B, self.num_heads, N, 1, self.num_samples)
        
        attn = attn.masked_fill(~valid_mask_expanded, float('-inf'))

        # valid_mask_expanded: (B, nH, N, 1, S)
        # any_valid: (B, nH, N, 1, 1)
        any_valid = valid_mask_expanded.any(dim=-1, keepdim=True)
        
        # This makes softmax output uniform (1/S), but we will mask the result later anyway.
        
        attn_safe = torch.where(any_valid, attn, torch.zeros_like(attn))
        
        attn = attn_safe.softmax(dim=-1)
        
        attn = attn.masked_fill(~any_valid, 0.0)

        out = attn @ v
        
        out = out.squeeze(3).transpose(1, 2).reshape(B, H, W, C)
        
        out = self.o_proj(out).permute(0, 3, 1, 2)
        
        out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4).clamp(-1e4, 1e4)

        if return_matching_loss:
            return out, matching_loss
        else:
            return out
