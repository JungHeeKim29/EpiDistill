import numpy as np
import torch
from PIL import Image

from unidepth.models import UniDepthV2
from unidepth.utils import colorize, image_grid
from unidepth.utils.camera import Pinhole


def demo(model):
    import os
    from glob import glob
    images = sorted(glob('./datasets/testdata/zoomin_frames/*.png'))
    images = sorted(glob('./datasets/zoom_dataset/train/00001/*.JPG'))
    for img in images:
        rgb = np.array(Image.open(img).convert("RGB")).astype(float) / 255.0
        rgb_torch = torch.from_numpy(rgb).permute(2, 0, 1)

        
        # intrinsics_torch = torch.from_numpy(np.load("assets/demo/intrinsics.npy"))
        # camera = Pinhole(K=intrinsics_torch.unsqueeze(0))
        
        # # infer method of V1 uses still the K matrix as input
        # if isinstance(model, (UniDepthV2old, UniDepthV1)):
        #     camera = camera.K.squeeze(0)

        # # predict
        # predictions = model.infer(rgb_torch, camera)
        predictions = model.infer(rgb_torch)
        
        # get GT and pred
        depth_pred = predictions["depth"].squeeze().cpu().numpy()
        intrinsics = predictions["intrinsics"].squeeze().cpu().numpy()
        # depth_gt = np.array(Image.open("assets/demo/depth.png")).astype(float) / 1000.0

        # compute error, you have zero divison where depth_gt == 0.0
        # depth_arel = np.abs(depth_gt - depth_pred) / depth_gt
        # depth_arel[depth_gt == 0.0] = 0.0

        # colorize
        # depth_pred_col = colorize(depth_pred, vmin=0.01, vmax=10.0, cmap="magma_r")
        # depth_gt_col = colorize(depth_gt, vmin=0.01, vmax=10.0, cmap="magma_r")
        # depth_error_col = colorize(depth_arel, vmin=0.0, vmax=0.2, cmap="coolwarm")

        # save image with pred and error
        # artifact = image_grid([rgb, depth_gt_col, depth_pred_col, depth_error_col], 2, 2)
        # Image.fromarray(artifact).save("assets/demo/output.png")

        img_name = img.split('/')[-1]
        out_dir = './datasets/testdata/zoomin_out_depthmap'
        out_int_dir = './datasets/testdata/zoomin_out_intrinsics'
        os.makedirs(out_dir, exist_ok=True)   
        os.makedirs(out_int_dir, exist_ok=True)   

        depth_pred = depth_pred * 1000.0  # convert to mm
        depth_pred = depth_pred.astype(np.uint16)

        Image.fromarray(depth_pred).save(os.path.join(out_dir, img_name))
        np.save(os.path.join(out_int_dir, img_name.replace('.png', '_intrinsics.npy')), intrinsics)

        # print("Available predictions:", list(predictions.keys()))
        # print(f"ARel: {depth_arel[depth_gt > 0].mean() * 100:.2f}%")


if __name__ == "__main__":
    
    print("Torch version:", torch.__version__)
    type_ = "l"  # available types: s, b, l
    name = f"unidepth-v2-vit{type_}14"
    model = UniDepthV2.from_pretrained(f"lpiccinelli/{name}")

    # set resolution level (only V2)
    model.resolution_level = 9

    # set interpolation mode (only V2)
    model.interpolation_mode = "bilinear"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device).eval()

    demo(model)
