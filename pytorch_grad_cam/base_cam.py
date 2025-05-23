from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
import ttach as tta

from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradients
from pytorch_grad_cam.utils.image import scale_cam_image
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.svd_on_activations import get_2d_projection
import torch.nn.functional as F
import os
import cv2
def save_topk_activations_with_input(input_tensor: torch.Tensor,
                                     activations: torch.Tensor,
                                     weights: np.ndarray,
                                     save_path: str,
                                     idx: int = 0,
                                     topk: int = 5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # 1. Input image (normalize to 0-255)
    input_img = input_tensor[idx].cpu().numpy().transpose(1, 2, 0)  # (224, 224, 3)
    input_img -= input_img.min()
    input_img /= (input_img.max() + 1e-8)
    input_img = (input_img * 255).astype(np.uint8)

    # [✔️ Fix] Convert to BGR to match heatmap color format
    input_img = cv2.cvtColor(input_img, cv2.COLOR_RGB2BGR)

    # 2. Activations & weights
    acts = activations[idx] # (C, 14, 14)
    w = weights[idx]  # shape: (C,)
    # 3. Get top-K indices by absolute weight (or just weight if no negative values)
    topk_indices = np.argsort(w)[-topk:][::-1]  # descending
    heatmaps = []
    for i in topk_indices:
        act_map = acts[i]  # (14, 14)
        act_map -= act_map.min()
        act_map /= (act_map.max() + 1e-8)
        act_map = np.uint8(act_map * 255)
        act_map_resized = cv2.resize(act_map, (224, 224), interpolation=cv2.INTER_LINEAR)
        heatmap = cv2.applyColorMap(act_map_resized, cv2.COLORMAP_JET)
        heatmaps.append(heatmap)
    # 4. Combine input image + top-K heatmaps horizontally
    all_imgs = [input_img] + heatmaps
    combined = np.concatenate(all_imgs, axis=1)  # side-by-side (224, (K+1)*224, 3)
    # 5. Save
    cv2.imwrite(save_path, combined)
    print(f":흰색_확인_표시: Top-{topk} activation heatmaps saved to: {save_path}")

def save_blended_cam(input_tensor: torch.Tensor,
                     activations: torch.Tensor,
                     weights: np.ndarray,
                     save_path: str,
                     idx: int = 0,
                     use_rgb: bool = True,
                     colormap: int = cv2.COLORMAP_JET,
                     image_weight: float = 0.5):
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    # 1. Prepare input image
    input_img = input_tensor[idx].cpu().numpy().transpose(1, 2, 0)  # (224, 224, 3)
    input_img -= input_img.min()
    input_img /= (input_img.max() + 1e-8)
    input_img = input_img.astype(np.float32)
    # 2. Prepare activations and weights
    acts = activations[idx]# (C, H, W)
    w = weights[idx]  # shape: (C,)
    # 3. Weighted sum → CAM (shape: H, W)
    cam = np.sum(w[:, None, None] * acts, axis=0)
    cam -= cam.min()
    cam /= (cam.max() + 1e-8)  # normalize to [0, 1]
    # 4. Resize CAM to input size
    cam_resized = cv2.resize(cam, (224, 224))
    heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), colormap)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB) if use_rgb else heatmap
    heatmap = np.float32(heatmap) / 255
    # 5. Blend
    blended = image_weight * input_img + (1 - image_weight) * heatmap
    blended = blended / np.max(blended)
    blended = np.uint8(255 * blended)
    # 6. Save
    cv2.imwrite(save_path, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR) if use_rgb else blended)
    print(f":흰색_확인_표시: Blended CAM image saved to: {save_path}")


class BaseCAM:    
    def __init__(
        self,
        model: torch.nn.Module,
        target_layers: List[torch.nn.Module],
        reshape_transform: Callable = None,
        compute_input_gradient: bool = False,
        uses_gradients: bool = True,
        tta_transforms: Optional[tta.Compose] = None,
        detach: bool = True, 
        device: Optional[torch.device] = None,
    ) -> None:
        self.model = model.eval()
        self.target_layers = target_layers

        # Use the same device as the model.
        self.device = device # next(self.model.parameters()).device
        
        #if 'hpu' in str(self.device):
            # try:
            #     import habana_frameworks.torch.core as htcore
            # except ImportError as error:
            #     error.msg = f"Could not import habana_frameworks.torch.core. {error.msg}."
            #     raise error
            #self.__htcore = htcore
        
        self.reshape_transform = reshape_transform
        self.compute_input_gradient = compute_input_gradient
        self.uses_gradients = uses_gradients
        if tta_transforms is None:
            self.tta_transforms = tta.Compose(
                [
                    tta.HorizontalFlip(),
                    tta.Multiply(factors=[0.9, 1, 1.1]),
                ]
            )
        else:
            self.tta_transforms = tta_transforms

        self.detach = detach
        self.activations_and_grads = ActivationsAndGradients(self.model, target_layers, reshape_transform, self.detach)

    """ Get a vector of weights for every channel in the target layer.
        Methods that return weights channels,
        will typically need to only implement this function. """

    def get_cam_weights(
        self,
        input_tensor: torch.Tensor,
        target_layers: List[torch.nn.Module],
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor,
    ) -> np.ndarray:
        raise Exception("Not Implemented")

    def get_cam_image(
        self,
        input_tensor: torch.Tensor, # batch_size x 3 x 224 x 224
        target_layer: torch.nn.Module, # LayerNorm((768,), eps=1e-06, elementwise_affine=True) / ReLU(inplace=True)
        targets: List[torch.nn.Module],
        activations: torch.Tensor,
        grads: torch.Tensor, # batch_size x 192 x 14 x 14 (vit) / (32, 512, 14, 14)(vgg16)
        eigen_smooth: bool = False,
        truncate_weight: bool = False
    ) -> np.ndarray:
        weights = self.get_cam_weights(input_tensor, target_layer, targets, activations, grads) # 32 x 192 (vit)
        
        # Negative Weight Clamping
        if truncate_weight:
            weights = np.maximum(weights, 0.0)

        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().detach().numpy()
        # 2D conv
        if len(activations.shape) == 4:
            weighted_activations = weights[:, :, None, None] * activations
        # 3D conv
        elif len(activations.shape) == 5:
            weighted_activations = weights[:, :, None, None, None] * activations
        else:
            raise ValueError(f"Invalid activation shape. Get {len(activations.shape)}.")

        if eigen_smooth:
            cam = get_2d_projection(weighted_activations)
        else:
            cam = weighted_activations.sum(axis=1)
        return cam

    def forward(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False, truncate_weight:bool = False
    ) -> np.ndarray:

        input_tensor = input_tensor.to(self.device)

        if self.compute_input_gradient:
            input_tensor = torch.autograd.Variable(input_tensor, requires_grad=True)

        if isinstance(self.activations_and_grads(input_tensor), dict):
            if 'logits2' in self.activations_and_grads(input_tensor).keys():
                outputs = self.activations_and_grads(input_tensor)['logits2']
                softmax_outputs = self.activations_and_grads(input_tensor)['logits']
            else:
                outputs = self.activations_and_grads(input_tensor)['logits']
                softmax_outputs = outputs

        self.outputs = outputs

        targets_was_none= False 
        if targets is None: 
            targets_was_none = True
            #target_categories = np.argmax(outputs.cpu().data.numpy(), axis=-1)
            target_categories = np.argmax(softmax_outputs.cpu().data.numpy(), axis=-1)
            targets = [ClassifierOutputTarget(category) for category in target_categories]

        if self.uses_gradients:
            loss = sum([target(output) for target, output in zip(targets, outputs)])

            self.model.zero_grad()
            if self.detach:
                loss.backward(retain_graph=True)
            else:
                # keep the computational graph, create_graph = True is needed for hvp
                torch.autograd.grad(loss, input_tensor, retain_graph = True, create_graph = True)
                # When using the following loss.backward() method, a warning is raised: "UserWarning: Using backward() with create_graph=True will create a reference cycle"
                # loss.backward(retain_graph=True, create_graph=True)
            if 'hpu' in str(self.device):
                self.__htcore.mark_step()

        # In most of the saliency attribution papers, the saliency is
        # computed with a single target layer.
        # Commonly it is the last convolutional layer.
        # Here we support passing a list with multiple target layers.
        # It will compute the saliency image for every image,
        # and then aggregate them (with a default mean aggregation).
        # This gives you more flexibility in case you just want to
        # use all conv layers for example, all Batchnorm layers,
        # or something else.
        cam_per_layer = self.compute_cam_per_layer(input_tensor, targets, eigen_smooth, truncate_weight)
        
        if targets_was_none: 
            return self.aggregate_multi_layers(cam_per_layer), targets
        else:
            return self.aggregate_multi_layers(cam_per_layer)

    def get_target_width_height(self, input_tensor: torch.Tensor) -> Tuple[int, int]:
        if len(input_tensor.shape) == 4:
            width, height = input_tensor.size(-1), input_tensor.size(-2)
            return width, height
        elif len(input_tensor.shape) == 5:
            depth, width, height = input_tensor.size(-1), input_tensor.size(-2), input_tensor.size(-3)
            return depth, width, height
        else:
            raise ValueError("Invalid input_tensor shape. Only 2D or 3D images are supported.")

    def compute_cam_per_layer(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool, truncate_weight:bool=False
    ) -> np.ndarray:
        if self.detach:
            activations_list = [a.cpu().data.numpy() for a in self.activations_and_grads.activations]
            grads_list = [g.cpu().data.numpy() for g in self.activations_and_grads.gradients]
        else:
            activations_list = [a for a in self.activations_and_grads.activations]
            grads_list = [g for g in self.activations_and_grads.gradients]
        target_size = self.get_target_width_height(input_tensor)

        cam_per_target_layer = []
        # Loop over the saliency image from every layer
        for i in range(len(self.target_layers)):
            target_layer = self.target_layers[i]
            layer_activations = None
            layer_grads = None
            if i < len(activations_list):
                layer_activations = activations_list[i]
            if i < len(grads_list):
                layer_grads = grads_list[i]

            cam = self.get_cam_image(input_tensor, 
                                     target_layer, 
                                     targets, 
                                     layer_activations, 
                                     layer_grads, 
                                     eigen_smooth, 
                                     truncate_weight) #Alpha

            cam = np.maximum(cam, 0)  #ReLU
            scaled = scale_cam_image(cam, target_size) # 128 x 224 x 224
            cam_per_target_layer.append(scaled[:, None, :])

        return cam_per_target_layer

    def aggregate_multi_layers(self, cam_per_target_layer: np.ndarray) -> np.ndarray:
        cam_per_target_layer = np.concatenate(cam_per_target_layer, axis=1)
    
        cam_per_target_layer = np.maximum(cam_per_target_layer, 0)   # ReLU
        result = np.mean(cam_per_target_layer, axis=1)
        return scale_cam_image(result) 
        
    def forward_augmentation_smoothing(
        self, input_tensor: torch.Tensor, targets: List[torch.nn.Module], eigen_smooth: bool = False
    ) -> np.ndarray:
        cams = []
        for transform in self.tta_transforms:
            augmented_tensor = transform.augment_image(input_tensor)
            cam = self.forward(augmented_tensor, targets, eigen_smooth)

            # The ttach library expects a tensor of size BxCxHxW
            cam = cam[:, None, :, :]
            cam = torch.from_numpy(cam)
            cam = transform.deaugment_mask(cam)

            # Back to numpy float32, HxW
            cam = cam.numpy()
            cam = cam[:, 0, :, :]
            cams.append(cam)

        cam = np.mean(np.float32(cams), axis=0)
        return cam

    def __call__(
        self,
        input_tensor: torch.Tensor,
        targets: List[torch.nn.Module] = None,
        aug_smooth: bool = False,
        eigen_smooth: bool = False,
        truncate_weight: bool = False
    ) -> np.ndarray:
        # Smooth the CAM result with test time augmentation
        if aug_smooth is True:
            print(f'✔️ Augmentation smoothing is applied.')
            return self.forward_augmentation_smoothing(input_tensor, targets, eigen_smooth)

        return self.forward(input_tensor, targets, eigen_smooth, truncate_weight)

    def __del__(self):
        self.activations_and_grads.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self.activations_and_grads.release()
        if isinstance(exc_value, IndexError):
            # Handle IndexError here...
            print(f"An exception occurred in CAM with block: {exc_type}. Message: {exc_value}")
            return True
