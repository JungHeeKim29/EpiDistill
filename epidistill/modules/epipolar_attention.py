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
    # S = [ z,  0, -x]
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
        # Return identity-like F to avoid crash (epipolar lines will be meaningless but training survives)
        return torch.eye(3, device=K_ref.device, dtype=K_ref.dtype).unsqueeze(0).expand(K_ref.shape[0], -1, -1)
    
    R = T_ref2src[:, :3, :3]
    t = T_ref2src[:, :3, 3] # (B, 3)
    
    # Essential Matrix E = [t]_x R
    t_skew = skew_symmetric(t)
    E = t_skew @ R
    
    # Fundamental Matrix F = K_src^-T E K_ref^-1
    K_ref_inv = torch.inverse(K_ref)
    K_src_inv = torch.inverse(K_src)
    
    F = K_src_inv.transpose(1, 2) @ E @ K_ref_inv
    
    return F

def get_epipole(K, T_ref2src):
    """
    Compute epipole in Ref image for a Virtual Source Camera.
    """
    R = T_ref2src[:, :3, :3]
    t = T_ref2src[:, :3, 3:4]
    
    C_src_in_ref = - R.transpose(1, 2) @ t # (B, 3, 1)
    
    # Project to Image Plane: e = K * C_src
    e_homo = K @ C_src_in_ref # (B, 3, 1)
    
    # Normalize
    div = e_homo[:, 2, 0:1]
    # Robust division
    div = torch.where(div.abs() < 1e-5, torch.sign(div + 1e-9) * 1e-5, div)
    e = e_homo[:, :2, 0] / div # (B, 2)
    
    return e

def get_epipolar_lines_cross(K_ref, K_src, T_ref2src, p_ref):
    """
    Compute epipolar lines in Source Image for points p_ref in Ref Image.
    l = F p_ref (l: [a, b, c] -> ax + by + c = 0)
    """
    F = compute_fundamental_matrix(K_ref, K_src, T_ref2src) # (B, 3, 3)
    lines = F @ p_ref # (B, 3, N)
    
    # Normalize lines: ax + by + c = 0 is equivalent to k(ax + by + c) = 0
    # Normalize such that sqrt(a^2 + b^2) = 1 for numerical stability in intersection calc
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
    
    # Intersections with borders
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
               
    # Project to 1D
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
    
    # Handle invalid
    p_start[~valid_line] = 0.0
    p_end[~valid_line] = 0.0
    
    # Interpolate
    t = torch.linspace(0, 1, num_samples, device=device).view(1, 1, num_samples, 1)
    points = p_start.unsqueeze(2) * (1 - t) + p_end.unsqueeze(2) * t
    
    # Sanitize
    points = torch.nan_to_num(points, nan=0.0, posinf=1e4, neginf=-1e4)
    
    line_len = torch.norm(p_end - p_start, dim=-1)
    valid_line = valid_line & (line_len > 1.0)
    valid_samples = valid_line.unsqueeze(-1).expand(-1, -1, num_samples)
    
    return points, valid_samples

def sample_lines_guided(lines, H, W, num_samples, K_ref, K_src, T_ref2src, p_ref_homo, depth, depth_window=0.1):
    """
    Sample points along epipolar lines focusing on depth range.
    """
    B, _, N = lines.shape
    device = lines.device
    
    # 1. Get Image Intersection (Maximal Segment)
    p_img_start, p_img_end, valid_img, dx, dy = get_line_image_intersection(lines, H, W)
    
    # 2. Project Depth Interval
    d_min = depth * (1.0 - depth_window)
    d_max = depth * (1.0 + depth_window)
    d_min = torch.clamp(d_min, min=1e-3)
    d_max = torch.clamp(d_max, min=1e-3)
    
    # Project to Source
    K_ref_inv = torch.inverse(K_ref)
    Ray_ref = K_ref_inv @ p_ref_homo # (B, 3, N)
    R = T_ref2src[:, :3, :3]
    t = T_ref2src[:, :3, 3:4]
    Ray_rot = R @ Ray_ref
    
    def normalize_homo(p_homo):
        z = p_homo[:, 2:3, :]
        valid_z = z > 1e-3
        p = p_homo[:, :2, :] / (z + 1e-6)
        return p, valid_z.squeeze(1)

    # Project min/max depth
    # Note: Inverse depth relationship means d_min might map to one end and d_max to other depending on motion?
    # Actually, R*p*d + t. It's linear in d (along ray direction).
    # But projection to pixel is non-linear.
    
    P_min = Ray_rot * d_min.unsqueeze(1) + t
    P_max = Ray_rot * d_max.unsqueeze(1) + t
    
    p_d_min, valid_min = normalize_homo(K_src @ P_min) # (B, 2, N)
    p_d_max, valid_max = normalize_homo(K_src @ P_max) # (B, 2, N)
    
    # Transpose to (B, N, 2)
    p_d_min = p_d_min.permute(0, 2, 1)
    p_d_max = p_d_max.permute(0, 2, 1)
    
    # We have 2 points from depth. They define a segment (or ray segment).
    # Project all 4 points to 1D line coordinate for interval intersection
    # axis: (dx, dy)
    def proj_1d(p):
        return p[..., 0] * dx + p[..., 1] * dy
        
    t_img_start = proj_1d(p_img_start) # (B, N)
    t_img_end   = proj_1d(p_img_end)
    
    # Re-order image bounds so t_start <= t_end
    t_img_min = torch.min(t_img_start, t_img_end)
    t_img_max = torch.max(t_img_start, t_img_end)
    
    t_d1 = proj_1d(p_d_min)
    t_d2 = proj_1d(p_d_max)
    
    # Re-order depth bounds
    t_d_min = torch.min(t_d1, t_d2)
    t_d_max = torch.max(t_d1, t_d2)
    
    # Handle invalid depth projection (e.g. zero/negative depth in sparse GT)
    # If depth is invalid (< 1e-3), we should rely ONLY on image bounds (fallback to uniform)
    # We accomplish this by setting depth interval to [-inf, inf] for those invalid points
    
    is_valid_d_val = depth > 1e-3
    
    t_d_min = torch.where(is_valid_d_val, t_d_min, torch.tensor(-float('inf'), device=device))
    t_d_max = torch.where(is_valid_d_val, t_d_max, torch.tensor(float('inf'), device=device))
    
    # Also valid_depth flag from projection shouldn't kill the line if we modify the interval
    # If depth val was invalid, we don't care if p_d_min/max were valid projections (they are likely garbage)
    # So we override valid_depth to True for those cases (since we are ignoring the depth interval)
    
    valid_depth = torch.where(is_valid_d_val, valid_min & valid_max, torch.ones_like(valid_min, dtype=torch.bool))

    # Intersect Intervals [t_img_min, t_img_max] AND [t_d_min, t_d_max]
    t_start = torch.max(t_img_min, t_d_min)
    t_end   = torch.min(t_img_max, t_d_max)
    
    # Check emptiness
    valid_interval = (t_end > t_start) & valid_depth & valid_img
    
    # Fallback to full valid image line if depth intersection is empty/invalid
    # (Optional: Or assume bad depth -> no attention)
    # Let's fallback to image bounds to simulate "high uncertainty" if depth projection fails?
    # User asked to use GT depth. If GT is invalid/behind, we shouldn't attend?
    # But usually GT is valid.
    # If intersection is empty (depth range is outside image), valid_interval is False.
    
    # However, if t_start > t_end, we have 0 samples. 
    # Let's just clamp t_start/end to image bounds if we want to force valid.
    # But strictly, we should respect the depth.
    
    # Reconstruct points from t
    # p(t) = (x,y) -> not easy unless we have a point on line.
    
    # Better: Interpolate between the computed 2D points corresponding to t_start/t_end.
    # We know p_img_start corresponds to t_img_start/end (unordered).
    # We need a reference point on the line.
    # p_on_line = p_img_start.
    # direction vector v = (dx, dy)? No, line is ax+by+c=0. Normal is (a,b). Direction is (-b, a) = (dx, dy).
    # Yes.
    # So p(alpha) = p_ref + alpha * (dx, dy)
    
    # But we projected p onto (dx, dy).
    # t = p . d
    # p(t)?
    # Since |d|^2 might not be 1.
    # d = (-b, a). a^2+b^2=1? Yes, in get_epipolar_lines_cross we normalized it.
    # So |d|=1.
    # Then p(t) = p_origin + t * d  is NOT quite right because t includes p_origin offset.
    # Correct: p(t) = (t_start * d) ?? No.
    
    # We can just interpolate between the 2D points we found!
    # We have 4 candidates for endpoints: p_img_start, p_img_end, p_d_min, p_d_max.
    # We selected t limits.
    # We need to map t back to 2D.
    # Or just select the 2D points corresponding to t_start and t_end.
    
    # We know which points generated t_d_min/max and t_img_min/max.
    # t_d_min came from either p_d_min or p_d_max.
    # Let's just pick the points.
    
    # Simplified logic:
    # We have segment [p_img_min, p_img_max] (ordered by t)
    # We have segment [p_d_min, p_d_max] (ordered by t)
    # Intersection is [max(start), min(end)].
    
    # Sort Image Points
    mask_img_swap = t_img_start > t_img_end
    p_img_1 = torch.where(mask_img_swap.unsqueeze(-1), p_img_end, p_img_start)
    p_img_2 = torch.where(mask_img_swap.unsqueeze(-1), p_img_start, p_img_end)
    
    # Sort Depth Points
    mask_d_swap = t_d1 > t_d2
    p_d_1 = torch.where(mask_d_swap.unsqueeze(-1), p_d_max, p_d_min)
    p_d_2 = torch.where(mask_d_swap.unsqueeze(-1), p_d_min, p_d_max)
    
    # Intersect
    # Start = max(img_1, d_1) -> pick point with larger t
    mask_start_d = t_d_min > t_img_min
    p_start = torch.where(mask_start_d.unsqueeze(-1), p_d_1, p_img_1)
    
    # End = min(img_2, d_2) -> pick point with smaller t
    mask_end_d = t_d_max < t_img_max
    p_end = torch.where(mask_end_d.unsqueeze(-1), p_d_2, p_img_2)
    
    # Debug Prints (First batch, middle pixel)
    # if True:
    #     idx = N // 2
    #     print(f"DEBUG: t_img [{t_img_min[0, idx]:.2f}, {t_img_max[0, idx]:.2f}]")
    #     print(f"DEBUG: t_d   [{t_d_min[0, idx]:.2f}, {t_d_max[0, idx]:.2f}]")
    #     print(f"DEBUG: t_int [{torch.max(t_img_min, t_d_min)[0, idx]:.2f}, {torch.min(t_img_max, t_d_max)[0, idx]:.2f}]")

    # Check validity again (t_start < t_end)
    valid_final = valid_interval # (B, N)
    
    # Handle fully invalid
    p_start[~valid_final] = 0.0
    p_end[~valid_final] = 0.0
    
    # Interpolate
    t_samp = torch.linspace(0, 1, num_samples, device=device).view(1, 1, num_samples, 1)
    points = p_start.unsqueeze(2) * (1 - t_samp) + p_end.unsqueeze(2) * t_samp
    
    # Clamp/Sanitize
    points = torch.nan_to_num(points, nan=0.0, posinf=1e4, neginf=-1e4)
    
    valid_samples = valid_final.unsqueeze(-1).expand(-1, -1, num_samples)
    
    return points, valid_samples

def get_epipolar_sampling_grid(
    query_coords,
    K_ref,
    K_src,
    T_ref2src,
    img_shape, # (H_s, W_s)
    num_samples=32,
    depth=None,
    mode='cross',
    depth_window=2.0,
    #0.1
):
    """
    Generate sampling grid (B, H, W, S, 2) for Epipolar Attention.
    Values are normalized to [-1, 1] for grid_sample.
    """
    if query_coords.dim() == 4:
        B, H, W, _ = query_coords.shape
        N = H * W
    else:
        # Assuming (B, N, 2)
        B, N, _ = query_coords.shape
        H = img_shape[0] if img_shape else int(math.sqrt(N))
        W = img_shape[1] if img_shape else int(math.sqrt(N))
    device = query_coords.device
    
    if mode == 'cross':
        # Flatten query coords
        p_ref_pix = query_coords.view(B, H*W, 2).permute(0, 2, 1) # (B, 2, N)
        
        # Homogeneous coordinates
        ones = torch.ones(B, 1, H*W, device=device)
        p_ref = torch.cat([p_ref_pix, ones], dim=1) # (B, 3, N)
        
        # Compute Epipolar Lines passing through Source Image
        lines = get_epipolar_lines_cross(K_ref, K_src, T_ref2src, p_ref) # (B, 3, N)
        
        if depth is not None:
             if depth.dim() == 4: # (B, 1, H, W)
                 depth_flat = depth.view(B, -1) # (B, N)
             elif depth.dim() == 3: # (B, H, W)
                 depth_flat = depth.view(B, -1)
             else:
                 depth_flat = depth # Assume acceptable shape

             # Always use uniform sampling across full epipolar line
             # Depth-biased attention (apply_depth_bias) handles focusing on GT projection
             sampling_points_pix, valid_samples = sample_lines_uniform(lines, img_shape[0], img_shape[1], num_samples)
        else:
            sampling_points_pix, valid_samples = sample_lines_uniform(lines, img_shape[0], img_shape[1], num_samples) # (B, N, S, 2)
        grid = sampling_points_pix
        
    elif mode == 'self':
        # Placeholder for self-attention epipole logic provided in original
        # For robustness, fallback to identity sampling if not strictly required or use similar logic if needed.
        # Assuming similar logic but generating lines from Epipole -> Pixel
        # (This path was not active in crash reports, focusing cleanup on 'cross')
        raise NotImplementedError("Self-attention mode cleanup pending.")

    # Reshape to spatial (B, H, W, S, 2)
    grid = grid.view(B, H, W, num_samples, 2)
    
    # Normalize grid from pixel coordinates [0, W-1] to [-1, 1]
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
        
        self.last_attn = None
        self.last_grid = None
        self.attn_history = []
        

    def forward(
        self,
        query_feat, # (B, C, H, W)
        key_feat,   # (B, C, H_s, W_s)
        value_feat, # (B, C, H_s, W_s)
        K_ref,      # (B, 3, 3)
        K_src,      # (B, 3, 3)
        T_ref2src,  # (B, 4, 4)
        depth=None,
        mode='cross',
        rectified=False, # New argument
        pos=None, # Position Encoding
        apply_depth_bias=False, # New argument: Bias attention towards projected depth
        return_matching_loss=False, # Return the cross-entropy attention loss (Eq. 5)
    ):
        if self.grad_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl,
                query_feat, key_feat, value_feat,
                K_ref, K_src, T_ref2src, depth, mode, rectified, pos, apply_depth_bias, return_matching_loss,
                use_reentrant=False
            )
        else:
            return self._forward_impl(query_feat, key_feat, value_feat, K_ref, K_src, T_ref2src, depth, mode, rectified, pos, apply_depth_bias, return_matching_loss)

    def _forward_impl(self, query_feat, key_feat, value_feat, K_ref, K_src, T_ref2src, depth, mode, rectified, pos, apply_depth_bias, return_matching_loss):
        B, C, H, W = query_feat.shape
        H_s, W_s = key_feat.shape[-2:]
        
        # 1. Generate Sampling Grid
        if rectified:
            # Rectified Stereo Sampling (Horizontal)
            # Sample uniformly along the same row (y)
            # x_s can be [0, W_s-1]
            
            # Create grid (B, H, W, S, 2)
            # grid[..., 0] = linspace(-1, 1)
            # grid[..., 1] = y_norm
            
            # Y coords normalized: [-1, 1]
            # y corresponds to query y.
            # query y ranges [0, H-1]. Normalized: 2*y/(H-1) - 1.
            
            # y_grid: (B, H, W, S)
            y_norm = torch.linspace(-1, 1, steps=H, device=query_feat.device).view(1, H, 1, 1).expand(B, H, W, self.num_samples)
            
            # x_grid: (B, H, W, S) -> -1 to 1
            x_norm = torch.linspace(-1, 1, steps=self.num_samples, device=query_feat.device).view(1, 1, 1, self.num_samples).expand(B, H, W, self.num_samples)
            
            sampling_grid = torch.stack([x_norm, y_norm], dim=-1) # (B, H, W, S, 2)
            
            # All valid for rectified logic
            valid_mask = torch.ones(B, H, W, self.num_samples, dtype=torch.bool, device=sampling_grid.device)
            
            
        else:
            # Epipolar Sampling
            # Create pixel coordinates for Query
            y, x = torch.meshgrid(torch.arange(H, device=query_feat.device), torch.arange(W, device=query_feat.device), indexing='ij')
            query_coords = torch.stack([x, y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1).float() # (B, H, W, 2)
            
            sampling_grid, valid_mask = get_epipolar_sampling_grid(
                query_coords,
                K_ref, K_src, T_ref2src,
                img_shape=(H_s, W_s),
                num_samples=self.num_samples,
                depth=depth,
                mode=mode
            ) # (B, H, W, S, 2)
        
        # Robustness: NaN check and Fix
        if torch.isnan(sampling_grid).any():
             # print("⚠️ Sampling Grid has NaN!")
             sampling_grid = torch.nan_to_num(sampling_grid)

        N = H * W
        flat_grid = sampling_grid.view(B, N, self.num_samples, 2)
        
        # Clamp grid for stability (grid_sample can handle outside, but gradients might be huge)
        flat_grid = flat_grid.clamp(-2.0, 2.0)

        # 2. Sample Keys and Values
        # grid output: (B, N, S, 2) -> grid_sample -> (B, C, N, S)
        sampled_k = F.grid_sample(key_feat, flat_grid, align_corners=True, mode='bilinear', padding_mode='zeros')
        sampled_v = F.grid_sample(value_feat, flat_grid, align_corners=True, mode='bilinear', padding_mode='zeros')
        
        # 3. Attention Computation
        # Q: (B, C, H, W) -> (B, H*W, C)
        q = self.q_proj(query_feat.permute(0, 2, 3, 1)).view(B, N, C)
        
        # K, V: (B, C, N, S) -> (B, N, S, C)
        k = self.k_proj(sampled_k.permute(0, 2, 3, 1)) 
        v = self.v_proj(sampled_v.permute(0, 2, 3, 1))
        
        # Multi-head reshape
        head_dim = self.dim // self.num_heads
        
        # q: (B, N, nH, d)
        q = q.view(B, N, self.num_heads, head_dim).transpose(1, 2) # (B, nH, N, d)
        
        # k, v: (B, N, S, nH, d) -> (B, nH, N, d, S) / (B, nH, N, S, d)
        k_reshaped = k.view(B, N, self.num_samples, self.num_heads, head_dim).permute(0, 3, 1, 2, 4) # (B, nH, N, S, d)
        v = v.view(B, N, self.num_samples, self.num_heads, head_dim).permute(0, 3, 1, 2, 4) # (B, nH, N, S, d)
        
        # --- Apply RoPE for Geometric Bias ---
        if self.rope is not None and pos is not None:
            # q: (B, nH, N, d). pos: (B, N, 2)
            q = self.rope(q, pos)
            
            if rectified:
                # Key Position Generation for Rectified Grid
                # Query pos is (B, N, 2) -> (B, H*W, 2) -> (y, x)
                # We need pos_k for keys (B, nH, N, S, d)
                # Key positions: y is same as Query y. x is 0..S-1.
                
                # Extract Y coords from query pos
                q_y = pos[..., 0].long() # (B, N)
                
                # Generate S indices 0..S-1
                s_idx = torch.arange(self.num_samples, device=q.device).unsqueeze(0).unsqueeze(0).expand(B, N, -1) # (B, N, S)
                
                # Expand Y to S
                k_y = q_y.unsqueeze(-1).expand(-1, -1, self.num_samples) # (B, N, S)
                
                # Combine to (B, N, S, 2) - Flatten to (B, N*S, 2) for RoPE
                pos_k = torch.stack([k_y, s_idx], dim=-1).view(B, N*self.num_samples, 2)
                
                # Flatten k to match: (B, nH, N*S, d)
                k_flat = k_reshaped.permute(0, 1, 2, 4, 3).flatten(2, 3) # (B, nH, N*S, d) 
                # Note: k_reshaped was (B, nH, N, S, d). Flattening N, S maps to N*S
                
                # Apply RoPE
                k_rope = self.rope(k_flat, pos_k)
                
                # Reshape back to (B, nH, N, S, d)
                # Then permute to (B, nH, N, d, S) for attention
                k = k_rope.view(B, self.num_heads, N, self.num_samples, head_dim).permute(0, 1, 2, 4, 3) # (B, nH, N, d, S)
                
            else:
                # Teacher / Epipolar Mode
                # Grid contains (x, y) sampling coordinates in [-1, 1]
                # We need to map these to pixel coordinates for RoPE
                
                # flat_grid: (B, N, S, 2)
                # Denormalize to [0, W-1] and [0, H-1]
                # x_pix = (x_norm + 1) * (W - 1) / 2
                
                grid_norm = flat_grid # (B, N, S, 2)
                
                x_norm = grid_norm[..., 0]
                y_norm = grid_norm[..., 1]
                
                # Use W_s, H_s from key_feat
                x_pix = (x_norm + 1.0) * (W_s - 1.0) / 2.0
                y_pix = (y_norm + 1.0) * (H_s - 1.0) / 2.0
                
                # Round to nearest integer for embedding lookup
                x_idx = x_pix.round().long().clamp(0, W_s - 1)
                y_idx = y_pix.round().long().clamp(0, H_s - 1)
                
                # Combine to (B, N*S, 2)
                # We need to match k_flat shape for RoPE
                # k_reshaped: (B, nH, N, d, S) -> permute to (B, nH, N, S, d) -> flatten N, S
                
                # pos_k: (B, N, S, 2) -> flatten to (B, N*S, 2)
                pos_k = torch.stack([y_idx, x_idx], dim=-1).view(B, N*self.num_samples, 2)
                
                # Flatten k: (B, nH, N*S, d)
                k_flat = k_reshaped.permute(0, 1, 2, 4, 3).flatten(2, 3)
                
                # Apply RoPE
                k_rope = self.rope(k_flat, pos_k)
                
                # Reshape back to (B, nH, N, S, d) -> permute to (B, nH, N, d, S)
                k = k_rope.view(B, self.num_heads, N, self.num_samples, head_dim).permute(0, 1, 2, 4, 3)

        else:
             k = k_reshaped.permute(0, 1, 2, 4, 3)
        
        # Q expension: (B, nH, N, 1, d)
        q = q.unsqueeze(3)
        
        # Attention Scores: (B, nH, N, 1, d) @ (B, nH, N, d, S) -> (B, nH, N, 1, S)
        attn = (q @ k) * self.scale

        matching_loss = None
        # --- GT Depth Bias Injection ---
        # Guard: skip depth bias entirely if K_ref is singular (near-zero fx/fy)
        _k_ref_ok = (K_ref.diagonal(dim1=-2, dim2=-1)[:, :2].abs() > 1e-6).all()
        if apply_depth_bias and depth is not None and mode == 'cross' and not rectified and _k_ref_ok:
             # Calculate Bias based on distance to Projected GT Depth
             # sampling_grid: (B, H, W, S, 2) in Normalized Coords [-1, 1]
             # depth: (B, H*W) or (B, H, W)
             
             # 1. Project Reference + Depth -> Source Pixel
             if depth.dim() == 3: depth_flat = depth.view(B, -1)
             elif depth.dim() == 4: depth_flat = depth.view(B, -1)
             else: depth_flat = depth
             
             # Reuse helper logic or reimplement for efficiency
             # We need p_ref coords.
             # p_ref_pix was generated in get_epipolar_sampling_grid but not returned.
             # Regenerate p_ref_pix
             y_ref, x_ref = torch.meshgrid(torch.arange(H, device=query_feat.device), torch.arange(W, device=query_feat.device), indexing='ij')
             p_ref_pix = torch.stack([x_ref, y_ref], dim=-1).view(1, N, 2).expand(B, N, 2).transpose(1, 2) # (B, 2, N)
             
             ones = torch.ones(B, 1, N, device=query_feat.device)
             p_ref_homo = torch.cat([p_ref_pix, ones], dim=1) # (B, 3, N)
             
             # Project to Source
             K_ref_inv = torch.inverse(K_ref)
             Ray_ref = K_ref_inv @ p_ref_homo 
             R = T_ref2src[:, :3, :3]
             t = T_ref2src[:, :3, 3:4]
             
             P_src = (R @ Ray_ref) * depth_flat.unsqueeze(1) + t # (B, 3, N)
             p_src_homo = K_src @ P_src
             
             z_src = p_src_homo[:, 2:3, :]
             valid_proj = z_src > 1e-3
             
             p_src_pix = p_src_homo[:, :2, :] / (z_src + 1e-6) # (B, 2, N)
             
             # Normalize to [-1, 1] using H_s, W_s
             # p_src_norm_x = 2 * p_src_pix_x / (W_s - 1) - 1
             # p_src_norm_y = 2 * p_src_pix_y / (H_s - 1) - 1
             
             p_src_norm = torch.zeros_like(p_src_pix)
             p_src_norm[:, 0, :] = 2.0 * p_src_pix[:, 0, :] / max(W_s - 1, 1) - 1.0
             p_src_norm[:, 1, :] = 2.0 * p_src_pix[:, 1, :] / max(H_s - 1, 1) - 1.0
             
             # p_src_norm: (B, 2, N) -> (B, N, 1, 2)
             target_points = p_src_norm.transpose(1, 2).unsqueeze(2) # (B, N, 1, 2)
             
             # flat_grid: (B, N, S, 2)
             # Compute Distance Squared: (B, N, S)
             dist_sq = ((flat_grid - target_points)**2).sum(dim=-1)
             
             # Gaussian-like Bias: -scale * dist_sq
             # We want high attention (0 bias) at dist=0, and low attention (-inf bias) far away.
             # Gaussian: exp(-dist^2 / sigma^2). in Log-space: -dist^2 / sigma^2.
             # Let sigma be small (e.g. 0.1 in norm coords ~ 5-10% of image). 
             # 1/sigma^2 ~ 100.
             
             bias_scale = 50.0
             attn_bias = -bias_scale * dist_sq # (B, N, S)
             
             # Handle invalid projection (z < 0) -> No bias (or huge negative bias?)
             # If projected point  s invalid, we shouldn't bias towards it.
             # valid_proj: (B, 1, N) -> (B, N, 1)
             valid_proj_flat = valid_proj.squeeze(1).unsqueeze(-1) # (B, N, 1)
             
             # If invalid, bias = 0.0 (Neutral) or maintain original behavior
             attn_bias = torch.where(valid_proj_flat, attn_bias, torch.zeros_like(attn_bias))

             if return_matching_loss:
                 # Calculate matching loss using CrossEntropy before injecting bias
                 # Target is the index with the maximum GT bias (smallest distance)
                 # Only consider valid projections and valid epipolar line samples
                 # attn: (B, nH, N, 1, S)
                 # attn_bias: (B, N, S)
                 # valid_mask: (B, H, W, S) -> (B, N, S)
                 
                 valid_mask_flat = valid_mask.view(B, N, self.num_samples) # (B, N, S)
                 gt_target_idx = torch.argmax(attn_bias, dim=-1) # (B, N)
                 
                 # Pool attention logits across heads
                 # attn_mean = attn.mean(dim=1).squeeze(2) # (B, N, S)
                 # Wait, computing CE directly on raw logits gives gradients to q, k
                 attn_logits = attn.mean(dim=1).squeeze(2) # (B, N, S)
                 
                 # Cross Entropy Loss
                 ce_loss = F.cross_entropy(
                     attn_logits.reshape(-1, self.num_samples), 
                     gt_target_idx.reshape(-1), 
                     reduction='none'
                 ).reshape(B, N)
                 
                 # Mask out invalid projections and completely invalid lines
                 # A line is valid if it has at least one valid sample, but we also rely on valid_proj
                 any_valid = valid_mask_flat.any(dim=-1) # (B, N)
                 valid_matching = valid_proj_flat.squeeze(-1) & any_valid # (B, N)
                 
                 if valid_matching.sum() > 0:
                     matching_loss = (ce_loss * valid_matching.float()).sum() / valid_matching.float().sum()
                 else:
                     matching_loss = torch.tensor(0.0, device=query_feat.device)

             # Add to attn: (B, nH, N, 1, S)
             # Expand bias: (B, 1, N, 1, S)
             attn = attn + attn_bias.unsqueeze(1).unsqueeze(3)
        
        # Robustness: Clamp attention scores

        # attn = attn.clamp(min=-50, max=50) # Removed global clamp, will mask invalid instead
        
        # Apply Validity Mask
        # valid_mask: (B, H, W, S) -> flatten to (B, N, S)
        # We need to reshape valid_mask to match attn: (B, nH, N, 1, S)
        
        valid_mask_flat = valid_mask.view(B, N, self.num_samples) # (B, N, S)
        valid_mask_expanded = valid_mask_flat.unsqueeze(1).unsqueeze(3).expand(B, self.num_heads, N, 1, self.num_samples)
        
        # Set invalid scores to -inf
        attn = attn.masked_fill(~valid_mask_expanded, float('-inf'))

        # Safe Softmax handling
        # 1. Identify rows where ALL entries are -inf (i.e. all invalid)
        # valid_mask_expanded: (B, nH, N, 1, S)
        # any_valid: (B, nH, N, 1, 1)
        any_valid = valid_mask_expanded.any(dim=-1, keepdim=True)
        
        # 2. To prevent NaN in softmax, replace -inf with 0 for all-invalid rows TEMPORARILY
        # This makes softmax output uniform (1/S), but we will mask the result later anyway.
        # Ideally, we want the output Feature to be 0 for these pixels.
        
        # If all are -inf, max is -inf -> exp(x-max) = exp(0) = 1 -> sum = S -> 1/S.
        # But directly passing -inf to softmax usually results in nan if all are -inf.
        # Workaround: Set input to 0 where all invalid.
        attn_safe = torch.where(any_valid, attn, torch.zeros_like(attn))
        
        attn = attn_safe.softmax(dim=-1)
        
        # 3. Mask the OUTPUT of softmax to true 0.0 where no valid samples existed
        # If no valid samples, we don't want to attend to anything (uniform attention to garbage is bad).
        # We want the resulting value-aggregation to be 0.
        attn = attn.masked_fill(~any_valid, 0.0)

        # Store for visualization (detach to save memory/graph)
        # Store for visualization (detach to save memory/graph)
        # We store history to capture multiple calls (Teacher/Student, Src1/Src2)
        if not hasattr(self, 'attn_history'):
            self.attn_history = []
        
        # Limit history size to prevent memory leaks during long training if not cleared
        if len(self.attn_history) > 100:
            self.attn_history.pop(0)
            
        self.attn_history.append({
            'attn': attn.detach().cpu(),
            'grid': sampling_grid.detach().cpu(),
            'rectified': rectified
        })
        
        self.last_grid = sampling_grid.detach().cpu()
        self.last_attn = attn.detach().cpu()
        
        # Output: (B, nH, N, 1, S) @ (B, nH, N, S, d) -> (B, nH, N, 1, d)
        out = attn @ v
        
        # Reshape Output
        out = out.squeeze(3).transpose(1, 2).reshape(B, H, W, C)
        
        # Output Projection: (B, H, W, C) -> (B, C, H, W)
        out = self.o_proj(out).permute(0, 3, 1, 2)
        
        # Final Robustness
        # Hook for gradient clamping: prevent explosion in backward pass
        if out.requires_grad:
             # Handle case where grad can be None (e.g., when tensor is not connected to loss)
             def safe_grad_hook(grad):
                 if grad is None:
                     return None
                 return torch.nan_to_num(grad, nan=0.0, posinf=1e4, neginf=-1e4)
             out.register_hook(safe_grad_hook)

        if torch.isnan(out).any() or torch.isinf(out).any():
             out = torch.nan_to_num(out, nan=0.0, posinf=1e4, neginf=-1e4)
             out = torch.clamp(out, min=-1e4, max=1e4)

        if return_matching_loss:
            return out, matching_loss
        else:
            return out
