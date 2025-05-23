import torch
import torch.nn as nn
import timm
from .util import initialize_weights
import os 
from .util import remove_layer

__all__ = ['vit']


class ViTCam(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(ViTCam, self).__init__()
        self.features = features 
        self.model_structure = kwargs['model_structure']
        self.unfreeze_layer = kwargs['unfreeze_layer']
        
                
        self.head = nn.Linear(192, num_classes, bias=True)

        if self.model_structure == 'b2':
            self.head2 = nn.Linear(192, num_classes, bias=True)

        self.target_layers = [self.features.blocks[-1].norm1]

        if kwargs['init_weights']: 
            initialize_weights(self.modules(), init_mode='xavier')

        if self.unfreeze_layer != 'all':
            print(f'[Debug][vit.py] Freeze layers: {self.unfreeze_layer}')
            self.freeze_layers()

        if kwargs['debug']: 
            self.check_params() 
        
    def forward(self, x, labels=None):
        x = self.features(x)
        logits = self.head(x)
        results = {'logits': logits} 

        if self.model_structure == 'b2':
            logits_2 = self.head2(x)
            results['logits2'] = logits_2

        return results


    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def check_params(self):
        trainable_params_lst = []
        print(f'[Sanity Check] Trainable Parameters are following ...')
        for name, param in self.named_parameters():
            print(f"{name} >> {param.requires_grad}")
            if param.requires_grad:
                trainable_params_lst.append(name)
        print('----------------------------------------------------------')
        print(f"Trainable Parameters: {trainable_params_lst}")
        print(f"Total # of parameters: {self.count_parameters()}")
        print('----------------------------------------------------------') 

    def freeze_layers(self):
        for name, param in self.named_parameters():
            param.requires_grad = False  
            if self.unfreeze_layer in ['head', 'head2']:
                types = [self.unfreeze_layer + ".weight", self.unfreeze_layer + ".bias"] # fc.weight, fc.bias, fc2.weight, fc2.bias, ...
                if name in types: 
                    param.requires_grad = True

        for module in self.modules():
            if isinstance(module, nn.BatchNorm2d):
                if hasattr(module, 'weight'):
                    module.weight.requires_grad_(False)
                if hasattr(module, 'bias'):
                    module.bias.requires_grad_(False)
                module.eval()  
                module.track_running_stats = False
                module.affine = False 

def add_prefix_to_state_dict(state_dict, prefix):
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('head.') or k.startswith('head2.'):
            new_state_dict[k] = v  # head는 prefix 없이 유지
        else:
            new_state_dict[prefix + k] = v
    return new_state_dict

def load_pretrained_model(model, path=None, **kwargs):
    strict_rule = True 
    if path:
        state_dict = torch.load(os.path.join(path, 'vit_base_patch16_224.pth'))
    else:
        state_dict = torch.hub.load('facebookresearch/deit:main','deit_tiny_patch16_224', pretrained=True).state_dict()

    if kwargs['dataset_name'] != 'ILSVRC':
        state_dict = remove_layer(state_dict, 'head')
        strict_rule = False 

    elif kwargs['model_structure'] == 'b2':
        strict_rule=False

    state_dict = add_prefix_to_state_dict(state_dict, 'features.')

    print(f'[Debug] Load state dict as STRICT RULE = {strict_rule}')
    model.load_state_dict(state_dict, strict=strict_rule)
    return model 


def vit(architecture_type='cam', pretrained=False, **kwargs):
    layers = torch.hub.load('facebookresearch/deit:main','deit_tiny_patch16_224', pretrained=False)
    layers.reset_classifier(0) 
    model = ViTCam(layers, **kwargs)
    
    if pretrained:
        print(f'[Debug][vit.py] Load pretrained model from vit_base_patch16_224') 
        model = load_pretrained_model(model, path=None, **kwargs)
    return model