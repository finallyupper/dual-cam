import torch
import tqdm
from pytorch_grad_cam.base_cam import BaseCAM


class ScoreCAM(BaseCAM):
    def __init__(
            self,
            model,
            target_layers,
            reshape_transform=None,
            device='cuda:0'):
        super(ScoreCAM, self).__init__(model,
                                       target_layers,
                                       reshape_transform=reshape_transform,
                                       uses_gradients=False,
                                       device=device)
        self.model_structure = 'vanilla'

    def get_cam_weights(self,
                        input_tensor, # batch_size x 3 x 224 x 224
                        target_layer,
                        targets,
                        activations, # activation map : batch_size x 512 x 14 x 14 (vgg16)
                        grads):
        with torch.no_grad():
            upsample = torch.nn.UpsamplingBilinear2d(
                size=input_tensor.shape[-2:])
            activation_tensor = torch.from_numpy(activations)
            activation_tensor = activation_tensor.to(self.device)

            upsampled = upsample(activation_tensor) # batch_size x 512 x 224 x 224

            maxs = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).max(dim=-1)[0] # batch_size x 512
            mins = upsampled.view(upsampled.size(0),
                                  upsampled.size(1), -1).min(dim=-1)[0] # batch_size x 512

            # Normalize activation map (0~1)
            maxs, mins = maxs[:, :, None, None], mins[:, :, None, None]
            upsampled = (upsampled - mins) / (maxs - mins + 1e-8) # normalize the activation map

            # Hadamard Product
            input_tensors = input_tensor[:, None,:, :] * upsampled[:, :, None, :, :]

            if hasattr(self, "batch_size"):
                BATCH_SIZE = self.batch_size
            else:
                BATCH_SIZE = 16 

            scores = []
            for target, tensor in zip(targets, input_tensors): #tqdm.tqdm(zip(targets, input_tensors), total=len(targets), desc="Calculating scores"):
                for i in (range(0, tensor.size(0), BATCH_SIZE)): # tqdm.tqdm
                    batch = tensor[i: i + BATCH_SIZE, :]

                    model_output = self.model(batch)
                    if isinstance(model_output, dict):
                        if 'logits2' in model_output.keys():
                            model_output = model_output['logits2']
                            self.model_structure = 'b2'
                        else:
                            model_output = model_output['logits']

                    outputs = [target(o).cpu().item() for o in model_output]
                    scores.extend(outputs)
            scores = torch.Tensor(scores)
            scores = scores.view(activations.shape[0], activations.shape[1]) # batch_size x 512

            if  self.model_structure == 'b2':
                scores = torch.clamp(scores, min=0.0)

            weights = torch.nn.Softmax(dim=-1)(scores).numpy()
            return weights
