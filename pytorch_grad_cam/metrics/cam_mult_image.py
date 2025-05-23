import torch
import numpy as np
from typing import List, Callable
from pytorch_grad_cam.metrics.perturbation_confidence import PerturbationConfidenceMetric, PerturbationConfidenceMetric_noDiff, PerturbationConfidenceMetricAll
from sklearn.metrics import auc

def multiply_tensor_with_cam(input_tensor: torch.Tensor,
                             cam: torch.Tensor):
    """ Multiply an input tensor (after normalization)
        with a pixel attribution map
    """
    return input_tensor * cam

def delete_pixels(input_tensor: torch.Tensor,
                  cam: torch.Tensor,
                  percentages: list): 
    assert cam.dim() == 2, "CAM tensor should be 2D (H, W)"
    assert input_tensor.dim() == 3, "Image tensor should be 3D (C, H, W)"

    masked_images = []
    flattened_cam = cam.flatten()
    sorted_indices = torch.argsort(flattened_cam, descending=True)
    total_pixels = flattened_cam.numel()

    image_copy = input_tensor.clone().detach()

    for p in percentages:
        num_pixels_to_mask = int(total_pixels * (p / 100))
        mask_indices = sorted_indices[:num_pixels_to_mask]

        # Mask top activated pixels (set to 0)
        masked_image = image_copy.clone().view(3, -1)
        masked_image[:, mask_indices] = 0
        masked_image = masked_image.view(image_copy.size())
        masked_images.append(masked_image)

    return masked_images

class CamMultImageConfidenceChange(PerturbationConfidenceMetric):
    def __init__(self):
        super(CamMultImageConfidenceChange,
              self).__init__(multiply_tensor_with_cam)

class CamMultImageConfidenceChange_all(PerturbationConfidenceMetric_noDiff):
    def __init__(self):
        super(CamMultImageConfidenceChange_all,self).__init__(multiply_tensor_with_cam)


class DropInConfidence(CamMultImageConfidenceChange):
    def __init__(self, return_ratio=False):
        super(DropInConfidence, self).__init__()
        self.return_ratio = return_ratio # Add: used for average-drop(%)

    def __call__(self, *args, **kwargs):
        scores = super(DropInConfidence, self).__call__(*args, **kwargs)

        if self.return_ratio:
            diff_scores = scores[0] # (batch_size, )
            ori_scores = scores[1] # (batch_size, )
            diff_scores = -diff_scores

            ratio = (np.maximum(diff_scores, 0) / ori_scores) * 100.0
            return ratio
        else:
            scores = -scores 
        return np.maximum(scores, 0)


class IncreaseInConfidence(CamMultImageConfidenceChange):
    def __init__(self):
        super(IncreaseInConfidence, self).__init__()

    def __call__(self, *args, **kwargs):
        scores = super(IncreaseInConfidence, self).__call__(*args, **kwargs)
        return np.float32(scores > 0)


#Add: DropInConfidence + IncreaseInConfidence
class ConfidenceChange(CamMultImageConfidenceChange):
    def __init__(self,):
        super(ConfidenceChange, self).__init__()

    def __call__(self, *args, **kwargs):
        scores = super(ConfidenceChange, self).__call__(*args, **kwargs)
        ori_scores = scores[1] # (batch_size, )
        scores = scores[0] # = (scores_after_imputation) - (scores)

        # 1. Avearage drop (lower is better)
        diff_scores = -scores # = (scores) - (scores_after_imputation)
        ratio = (np.maximum(diff_scores, 0) / (ori_scores + 1e-7)) 

        # 2. Increase in confidence (higher is better)
        count = np.float32(scores > 0) 
        return ratio, count, ori_scores[0]

