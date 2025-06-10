"""
Copyright (c) 2020-present NAVER Corp.

Permission is hereby granted, free of charge, to any person obtaining a copy of
this software and associated documentation files (the "Software"), to deal in
the Software without restriction, including without limitation the rights to
use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of
the Software, and to permit persons to whom the Software is furnished to do so,
subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS
FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER
IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN
CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import json
import numpy as np
import os
import sys
import torch 
import matplotlib.pyplot as plt
import cv2

class Logger(object):
    """Log stdout messages."""

    def __init__(self, outfile):
        self.terminal = sys.stdout
        self.log = open(outfile, "w")
        sys.stdout = self
        self.log.write(f"[Command] {' '.join(sys.argv)}\n") #Add

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()


def t2n(t):
    return t.detach().cpu().numpy().astype(np.float32)

def check_scoremap_validity(scoremap):
    if not isinstance(scoremap, np.ndarray):
        raise TypeError("Scoremap must be a numpy array; it is {}."
                        .format(type(scoremap)))
    if scoremap.dtype != np.float32:
        raise TypeError("Scoremap must be of np.float32 type; it is of {} type."
                        .format(scoremap.dtype))
    if len(scoremap.shape) != 2:
        raise ValueError("Scoremap must be a 2D array; it is {}D."
                         .format(len(scoremap.shape)))
    if np.isnan(scoremap).any():
        raise ValueError("Scoremap must not contain nans.")
    if (scoremap > 1).any() or (scoremap < 0).any():
        raise ValueError("Scoremap must be in range [0, 1]."
                         "scoremap.min()={}, scoremap.max()={}."
                         .format(scoremap.min(), scoremap.max()))


def string_contains_any(string, substring_list):
    for substring in substring_list:
        if substring in string:
            return True
    return False


def convert_to_native(obj):
    if isinstance(obj, dict):
        return {k: convert_to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_native(v) for v in obj]
    elif isinstance(obj, np.generic):  # includes np.float32, np.int32, etc.
        return obj.item()
    else:
        return obj
    
class Reporter(object):
    def __init__(self, reporter_log_root, epoch):
        self.log_file = os.path.join(reporter_log_root, str(epoch))
        self.epoch = epoch
        self.report_dict = {
            'summary': True,
            'step': self.epoch,
        }

    def add(self, key, val):
        self.report_dict.update({key: val})

    def write(self):
        log_file = self.log_file
        while os.path.isfile(log_file):
            log_file += '_'
        with open(log_file, 'w') as f:
            native_dict = convert_to_native(self.report_dict)
            f.write(json.dumps(native_dict))


def check_box_convention(boxes, convention):
    """
    Args:
        boxes: numpy.ndarray(dtype=np.int or np.float32, shape=(num_boxes, 4))
        convention: string. One of ['x0y0x1y1', 'xywh'].
    Raises:
        RuntimeError if box does not meet the convention.
    """
    if (boxes < 0).any():
        raise RuntimeError("Box coordinates must be non-negative.")

    if len(boxes.shape) == 1:
        boxes = np.expand_dims(boxes, 0)
    elif len(boxes.shape) != 2:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if boxes.shape[1] != 4:
        raise RuntimeError("Box array must have dimension (4) or "
                           "(num_boxes, 4).")

    if convention == 'x0y0x1y1':
        widths = boxes[:, 2] - boxes[:, 0]
        heights = boxes[:, 3] - boxes[:, 1]
    elif convention == 'xywh':
        widths = boxes[:, 2]
        heights = boxes[:, 3]
    else:
        raise ValueError("Unknown convention {}.".format(convention))

    if (widths < 0).any() or (heights < 0).any():
        raise RuntimeError("Boxes do not follow the {} convention."
                           .format(convention))


class DataParallel(torch.nn.DataParallel):
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name) 
        

def plot_confidence_curves(deletion_confidence_dict, insertion_confidence_dict,save_dir):
    """
    Plot and save the deletion and insertion confidence curves.

    Args:
        deletion_confidence_dict (dict): Dictionary of deletion confidence values per percentile.
        insertion_confidence_dict (dict): Dictionary of insertion confidence values per percentile.
        save_path (str): Path to save the plot image.
    """
    plt.figure(figsize=(10, 6))

    # Plot deletion curve
    x = list(deletion_confidence_dict.keys())
    x = [float(i) for i in x]
    
    y_deletion = list(deletion_confidence_dict.values())
    
    plt.plot(x, y_deletion, label='gradcam', color='blue')

    plt.title("Deletion Curve")
    plt.xlabel("Percentage of Pixels")
    plt.ylabel("Confidence")
    plt.legend(loc="best")
    plt.savefig(os.path.join(save_dir, "deletion_curve.png"))
    plt.close()


    y_insertion = list(insertion_confidence_dict.values())
    plt.plot(x, y_insertion, label='gradcam', color='blue')
    plt.title("Insertion Curve")
    plt.xlabel("Percentage of Pixels")
    plt.ylabel("Confidence")
    plt.legend(loc="best")
    plt.savefig(os.path.join(save_dir, "insertion_curve.png"))
    plt.close()

    print("Confidence curves saved to {}".format(save_dir))

def unnormalize_image(images: torch.Tensor, mean, std,) -> torch.Tensor:
    """
    Unnormalize a batch of images.

    Args:
        images (torch.Tensor): Tensor of shape (B, C, H, W)
        mean (list or tuple): per-channel mean (e.g., [0.485, 0.456, 0.406])
        std (list or tuple): per-channel std (e.g., [0.229, 0.224, 0.225])

    Returns:
        torch.Tensor: Unnormalized image tensor
    """
    device = images.device
    mean = torch.tensor(mean, dtype=images.dtype, device=device).view(1, -1, 1, 1)
    std = torch.tensor(std, dtype=images.dtype, device=device).view(1, -1, 1, 1)
    return images * std + mean


def save_cam_blend_from_ready_cams(images: torch.Tensor,
                                   cams: np.ndarray,
                                   save_path: str,
                                   idx: int = 0,
                                   use_rgb: bool = True,
                                   colormap: int = cv2.COLORMAP_JET,
                                   image_weight: float = 0.5):
    """
    Save blended CAM+image and CAM heatmap (CAM-only) from (B, 224, 224) CAM maps.
    """

    # 1. Input image
    img = images[idx].cpu().numpy().transpose(1, 2, 0)  # (224, 224, 3)
    img -= img.min()
    img /= (img.max() + 1e-8)
    img = img.astype(np.float32)

    # 2. CAM
    cam = cams[idx]
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)

    # 3. Generate heatmap
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), colormap)
    if use_rgb:
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255

    # 4. Blend
    blended = image_weight * img + (1 - image_weight) * heatmap
    blended = blended / np.max(blended)
    blended = np.uint8(255 * blended)

    # 5. Save blended image
    if use_rgb:
        blended_bgr = cv2.cvtColor(blended, cv2.COLOR_RGB2BGR)
        heatmap_bgr = cv2.cvtColor(np.uint8(255 * heatmap), cv2.COLOR_RGB2BGR)
    else:
        blended_bgr = blended
        heatmap_bgr = np.uint8(255 * heatmap)

    cv2.imwrite(save_path, blended_bgr)

    # 6. Save CAM-only image
    cam_save_path = save_path.replace(".png", "_cam.png")
    cv2.imwrite(cam_save_path, heatmap_bgr)

    print(f"✅ Blended CAM image saved to: {save_path}")
    print(f"✅ CAM heatmap image saved to: {cam_save_path}")

import numpy as np
import os

def save_cam_weights(cam_weights, save_dir, prefix='cam_weights'):
    os.makedirs(save_dir, exist_ok=True)
    cam_weights_np = cam_weights.detach().cpu().numpy()  # (B, C)
    
    for i, weights in enumerate(cam_weights_np):
        save_path = os.path.join(save_dir, f"{prefix}_{i}.txt")
        np.savetxt(save_path, weights, fmt="%.6f")
        print(f"Saved: {save_path}")


def save_concatenated_cam(blend_path1, blend_path2, save_path):
    img1 = cv2.imread(blend_path1)
    img2 = cv2.imread(blend_path2)
    
    if img1 is None or img2 is None:
        print(f"❌ 이미지 로드 실패: {blend_path1}, {blend_path2}")
        return
    
    # Ensure images are same height
    if img1.shape[0] != img2.shape[0]:
        target_height = min(img1.shape[0], img2.shape[0])
        img1 = cv2.resize(img1, (int(img1.shape[1] * target_height / img1.shape[0]), target_height))
        img2 = cv2.resize(img2, (int(img2.shape[1] * target_height / img2.shape[0]), target_height))
    
    concat = np.concatenate([img1, img2], axis=1) 
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    cv2.imwrite(save_path, concat)
    print(f"✅ 저장됨: {save_path}")

# input_img: (B, 3, H, W), assuming normalized tensor
import torchvision.transforms.functional as TF
def save_top5_cam(feature_map, cam_weights, input_img, save_path, idx=0):
    os.makedirs(save_path, exist_ok=True)

    fmap = feature_map[idx]              # (C, H, W)
    weights = cam_weights[idx]           # (C,)
    image = input_img[idx]               # (3, H, W)
    
    top5_idx = torch.topk(weights, 5).indices  # Top-5 channel indices

    heatmaps = []
    for i, ch_idx in enumerate(top5_idx):
        cam = fmap[ch_idx] * weights[ch_idx]  # (H, W)
        cam = cam.detach().cpu().numpy()
        cam -= cam.min()
        cam /= (cam.max() + 1e-8)
        cam = np.uint8(255 * cam)
        heatmap = cv2.applyColorMap(cam, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        heatmaps.append(heatmap)

    # Convert input tensor to uint8 image
    img = image.detach().cpu()
    img = TF.normalize(img, [-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])  # Unnormalize
    img = TF.to_pil_image(img)
    img = np.array(img)

    # Concatenate input and top5 heatmaps
    concat = np.concatenate([img] + heatmaps, axis=1)
    out_path = os.path.join(save_path, f"top5_cam_{idx}.png")
    cv2.imwrite(out_path, cv2.cvtColor(concat, cv2.COLOR_RGB2BGR))
    print(f"Saved: {out_path}")


def save_selected_feature_maps_by_weight(feature_map, cam_weights, save_dir='./maps', prefix='img', resize=(224, 224)):
    """
    feature_map: [B, C, H, W]
    cam_weights: [B, C]
    """
    os.makedirs(save_dir, exist_ok=True)
    B, C, H, W = feature_map.shape

    for b in range(B):
        fmap = feature_map[b]            # [C, H, W]
        weights = cam_weights[b]         # [C]

        mask_neg = (weights > -0.05) & (weights < 0.0)
        mask_pos = (weights > 0.0) & (weights < 0.05)

        idx_neg = mask_neg.nonzero(as_tuple=False).squeeze(1)
        idx_pos = mask_pos.nonzero(as_tuple=False).squeeze(1)

        selected_neg = idx_neg[:10]
        selected_pos = idx_pos[:10]

        selected_channels = torch.cat([selected_neg, selected_pos], dim=0)

        img_dir = os.path.join(save_dir, f"{prefix}_{b}")

        import shutil
        if os.path.exists(img_dir):
            shutil.rmtree(img_dir)  # Remove the directory and all its contents
        os.makedirs(img_dir, exist_ok=True)  # Recreate the directory



        for i, ch in enumerate(selected_channels):
            ch = ch.item()
            w = weights[ch].item()
            fmap_ch = fmap[ch]

            fmap_weighted = fmap_ch * w
            fmap_np = fmap_weighted.detach().cpu().numpy()
            fmap_np -= fmap_np.min()
            fmap_np /= (fmap_np.max() + 1e-8)
            fmap_np = np.uint8(255 * fmap_np)
            fmap_resized = cv2.resize(fmap_np, resize, interpolation=cv2.INTER_LINEAR)
            heatmap = cv2.applyColorMap(fmap_resized, cv2.COLORMAP_JET)

            save_path = os.path.join(img_dir, f"{prefix}_{b}_idx{i}_ch{ch}_w{w:+.3f}.png")
            cv2.imwrite(save_path, heatmap)

            w_plus = w + 0.05
            fmap_weighted_plus = fmap_ch * w_plus
            fmap_np_plus = fmap_weighted_plus.detach().cpu().numpy()
            fmap_np_plus -= fmap_np_plus.min()
            fmap_np_plus /= (fmap_np_plus.max() + 1e-8)
            fmap_np_plus = np.uint8(255 * fmap_np_plus)
            fmap_resized_plus = cv2.resize(fmap_np_plus, resize, interpolation=cv2.INTER_LINEAR)
            heatmap_plus = cv2.applyColorMap(fmap_resized_plus, cv2.COLORMAP_JET)

            save_path_plus = os.path.join(img_dir, f"{prefix}_{b}_idx{i}_ch{ch}_w{w:+.3f}_plus0.05.png")
            cv2.imwrite(save_path_plus, heatmap_plus)


            fmap_np_orig = fmap_ch.detach().cpu().numpy()
            fmap_np_orig -= fmap_np_orig.min()
            fmap_np_orig /= (fmap_np_orig.max() + 1e-8)
            fmap_np_orig = np.uint8(255 * fmap_np_orig)
            fmap_resized_orig = cv2.resize(fmap_np_orig, resize, interpolation=cv2.INTER_LINEAR)
            heatmap_orig = cv2.applyColorMap(fmap_resized_orig, cv2.COLORMAP_JET)

            save_path_orig = os.path.join(img_dir, f"{prefix}_{b}_idx{i}_ch{ch}_original.png")
            cv2.imwrite(save_path_orig, heatmap_orig)
        
            print(f"✅ Saved: batch {b}, idx {i}, channel {ch}, w={w:.4f}")


import os
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as TF

def compute_iou(boxA, boxB):
    # box: [x1, y1, x2, y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    boxAArea = max(0, boxA[2] - boxA[0] + 1) * max(0, boxA[3] - boxA[1] + 1)
    boxBArea = max(0, boxB[2] - boxB[0] + 1) * max(0, boxB[3] - boxB[1] + 1)

    iou = interArea / float(boxAArea + boxBArea - interArea + 1e-6)
    return iou

def save_cam_with_image_and_bbox(image, cam, gt_bbox, image_id, save_path, loc_threshold=0.2, alpha=0.5):
    os.makedirs(save_path, exist_ok=True)

    # 1. Unnormalize the image
    img = image.detach().cpu()
    img = TF.normalize(img, [-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])
    img = TF.to_pil_image(img)
    img = np.array(img)
    H, W = img.shape[:2]

    # 2. CAM → heatmap
    cam_uint8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 3. Blend
    overlay = np.uint8(img * (1 - alpha) + heatmap * alpha)

    # 4. CAM → predicted bbox
    binary_mask = np.uint8(cam > loc_threshold) * 255
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    pred_bbox = None
    if contours:
        max_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(max_contour)
        pred_bbox = [x, y, x + w, y + h]

        # bbox_area = w * h
        # image_area = W * H
        # if bbox_area / image_area >= 0.8:
        #     return  

        # if gt_bbox is not None:
        #     gt_bbox = list(map(int, gt_bbox))
        #     iou = compute_iou(pred_bbox, gt_bbox)
        #     if iou > 0.8:
        #         return  

        # Draw predicted bbox (green)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

    # 5. Draw GT bbox (red)
    if gt_bbox is not None:
        x1, y1, x2, y2 = map(int, gt_bbox)
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

    # 6. Save
    out_path = os.path.join(save_path, image_id)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


# def save_cam_with_image_and_bbox(image, cam, gt_bbox, image_id, save_path, loc_threshold=0.2, alpha=0.5):
#     os.makedirs(save_path, exist_ok=True)

#     # 1. Unnormalize the image
#     img = image.detach().cpu()
#     img = TF.normalize(img, [-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225])  # Undo ImageNet normalization
#     img = TF.to_pil_image(img)
#     img = np.array(img)  # Convert to (H, W, 3) RGB uint8

#     # 2. Normalized CAM and apply colormap
#     # cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
#     cam_uint8 = np.uint8(255 * cam)
#     heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
#     heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

#     # 3. Blend
#     overlay = np.uint8(img * (1 - alpha) + heatmap * alpha)


#     binary_mask = np.uint8(cam > loc_threshold) * 255
#     contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#     pred_bbox = None
#     if contours:
#         max_contour = max(contours, key=cv2.contourArea)
#         x, y, w, h = cv2.boundingRect(max_contour)
#         pred_bbox = [x, y, x + w, y + h]
#         # Draw predicted bbox (green)
#         cv2.rectangle(overlay, (x, y), (x + w, y + h), color=(0, 255, 0), thickness=2)

#     # 5. Draw ground truth bbox (red)
#     if gt_bbox is not None:
#         x1, y1, x2, y2 = map(int, gt_bbox)
#         cv2.rectangle(overlay, (x1, y1), (x2, y2), color=(255, 0, 0), thickness=2)

#     # 4. Save
#     save_fname = os.path.join(save_path, image_id)
#     os.makedirs(os.path.dirname(save_fname), exist_ok=True)
#     out_path = os.path.join(save_path, image_id)
#     cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))


def save_cam_with_image(image, cam, image_id, save_path,alpha=0.5):
    os.makedirs(save_path, exist_ok=True)

    # 1. Unnormalize the image
    img = image.detach().cpu()
    img = TF.normalize(img, [-0.485/0.229, -0.456/0.224, -0.406/0.225], [1/0.229, 1/0.224, 1/0.225]) 
    img = TF.to_pil_image(img)
    img = np.array(img)  # Convert to (H, W, 3) RGB uint8

    # 2. Normalized CAM and apply colormap
    # cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    cam_uint8 = np.uint8(255 * cam)
    heatmap = cv2.applyColorMap(cam_uint8, cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

    # 3. Blend
    overlay = np.uint8(img * (1 - alpha) + heatmap * alpha)

    # 4. Save
    save_fname = os.path.join(save_path, image_id)
    os.makedirs(os.path.dirname(save_fname), exist_ok=True)
    out_path = os.path.join(save_path, image_id)
    cv2.imwrite(out_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
