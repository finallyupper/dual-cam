"""
Original code: https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.model_zoo import load_url

from .method import AcolBase
from .method import ADL
from .method import spg
from .method.util import normalize_tensor
from .util import remove_layer
from .util import replace_layer
from .util import initialize_weights

__all__ = ['vgg16']

model_urls = {
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth'
}

configs_dict = {
    'cam': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 'M', 512, 512, 512],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512,
                  512, 512, 512, 512],
    },
    'acol': {
        '14x14': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512,
                  512, 'M1', 512, 512, 512, 'M2'],
        '28x28': [64, 64, 'M1', 128, 128, 'M1', 256, 256, 256, 'M1', 512, 512,
                  512, 'M2', 512, 512, 512, 'M2'],
    },
    'spg': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M'],
    },
    'adl': {
        '14x14': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'A', 512,
                  512, 512, 'A', 'M', 512, 512, 512, 'A'],
        '28x28': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 'A', 512,
                  512, 512, 'A', 512, 512, 512, 'A'],
    }
}


class VggCam(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(VggCam, self).__init__()
        self.features = features
        self.conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.last_layer = kwargs['last_layer'] 
        self.unfreeze_layer = kwargs['unfreeze_layer']
        self.model_structure = kwargs['model_structure'] 

        if self.last_layer == 'fc':
            self.fc = nn.Linear(1024, num_classes) 
        else:
            self.conv = nn.Conv2d(1024, num_classes, kernel_size=1, padding=0)
        
        # branch 2 structure 
        if self.model_structure == 'b2': 
            if self.last_layer == 'fc':
                self.fc2 = nn.Linear(1024, num_classes)
            else:
                self.conv2 = nn.Conv2d(1024, num_classes, kernel_size=1, padding=0)

        if kwargs['init_weights']:
            initialize_weights(self.modules(), init_mode='he')
        if self.unfreeze_layer != 'all':
            self.freeze_layers() 
        
        if kwargs['debug']:
            self.check_params() 

    def forward(self, x, labels=None, return_cam=False):
        x = self.features(x)
        x = self.conv6(x)
        x = self.relu(x)
        pre_logit = self.avgpool(x)
        pre_logit = pre_logit.view(pre_logit.size(0), -1)

        if self.last_layer == 'fc':
            logits = self.fc(pre_logit)
        else:
            logits = self.conv(pre_logit) 
        results = {'logits': logits} 

        if return_cam:
            feature_map = x.detach().clone()

            if labels is None:
                labels = logits.argmax(dim=1)
                results['labels'] = labels

            cam_weights = self.fc.weight[labels]

            cams = (cam_weights.view(*feature_map.shape[:2], 1, 1) *feature_map).mean(1, keepdim=False)
            results['cams'] = cams 


        if self.model_structure == 'b2':
            if self.last_layer == 'fc':
                logits2 = self.fc2(pre_logit)
            else:
                logits2 = self.conv2(pre_logit)
            results['logits2'] = logits2

            if return_cam:
                feature_map = x.detach().clone()

                # if lables is None, use logits.argmax(dim=1) from head1
                cam_weights = self.fc2.weight[labels] 

                cam_weights = torch.where(torch.gt(cam_weights, 0.), cam_weights, torch.zeros_like(cam_weights))

                cams2 = (cam_weights.view(*feature_map.shape[:2], 1, 1) *
                        feature_map).mean(1, keepdim=False)
                results['cams2'] = cams2
                
        return results # {'logits': logits}

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
            if self.unfreeze_layer in ['fc', 'fc2', 'conv', 'conv2']:
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
                
class VggAcol(AcolBase):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(VggAcol, self).__init__()

        self.features = features
        self.drop_threshold = kwargs['acol_drop_threshold']

        self.classifier_A = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0),
        )
        self.classifier_B = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, 1024, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=False),
            nn.Conv2d(1024, num_classes, kernel_size=1, stride=1, padding=0),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(1)

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        feature = self.features(x)
        feature = F.avg_pool2d(feature, kernel_size=3, stride=1, padding=1)
        logits_dict = self._acol_logits(feature=feature, labels=labels,
                                        drop_threshold=self.drop_threshold)

        if return_cam:
            normalized_a = normalize_tensor(
                logits_dict['feat_map_a'].detach().clone())
            normalized_b = normalize_tensor(
                logits_dict['feat_map_b'].detach().clone())
            feature_map = torch.max(normalized_a, normalized_b)
            cams = feature_map[range(batch_size), labels]
            return cams

        return logits_dict


class VggSpg(nn.Module):
    def __init__(self, features, num_classes=1000, **kwargs):
        super(VggSpg, self).__init__()

        self.features = features
        self.lfs = kwargs['large_feature_map']

        self.SPG_A_1 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_A_2 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_A_3 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_A_4 = nn.Conv2d(512, num_classes, kernel_size=1, padding=0)

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.SPG_B_1a = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_B_2a = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.SPG_B_shared = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
        )

        self.SPG_C = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 1, kernel_size=1),
        )

        initialize_weights(self.modules(), init_mode='xavier')

    def forward(self, x, labels=None, return_cam=False):
        batch_size = x.shape[0]

        x = self.features(x)
        x = self.SPG_A_1(x)
        if not self.lfs:
            x = F.max_pool2d(x, kernel_size=3, stride=2, padding=1)

        logits_b1 = self.SPG_B_1a(x)
        logits_b1 = self.SPG_B_shared(logits_b1)

        x = self.SPG_A_2(x)
        logits_b2 = self.SPG_B_2a(x)
        logits_b2 = self.SPG_B_shared(logits_b2)

        x = self.SPG_A_3(x)
        logits_c = self.SPG_C(x)

        feat_map = self.SPG_A_4(x)
        logits = self.avgpool(feat_map)
        logits = logits.flatten(1)

        labels = logits.argmax(dim=1).long() if labels is None else labels
        attention, fused_attention = spg.compute_attention(
            feat_map=feat_map, labels=labels,
            logits_b1=logits_b1, logits_b2=logits_b2)

        if return_cam:
            feature_map = feat_map.clone().detach()
            cams = feature_map[range(batch_size), labels]
            return cams

        return {'attention': attention, 'fused_attention': fused_attention,
                'logits': logits, 'logits_b1': logits_b1,
                'logits_b2': logits_b2, 'logits_c': logits_c}


def adjust_pretrained_model(pretrained_model, current_model):
    def _get_keys(obj, split):
        keys = []
        iterator = obj.items() if split == 'pretrained' else obj
        for key, _ in iterator:
            if key.startswith('features.'):
                keys.append(int(key.strip().split('.')[1].strip()))
        return sorted(list(set(keys)), reverse=True)

    def _align_keys(obj, key1, key2):
        for suffix in ['.weight', '.bias']:
            old_key = 'features.' + str(key1) + suffix
            new_key = 'features.' + str(key2) + suffix
            obj[new_key] = obj.pop(old_key)
        return obj

    pretrained_keys = _get_keys(pretrained_model, 'pretrained')
    current_keys = _get_keys(current_model.named_parameters(), 'model')

    for p_key, c_key in zip(pretrained_keys, current_keys):
        pretrained_model = _align_keys(pretrained_model, p_key, c_key)

    return pretrained_model


def batch_replace_layer(state_dict):
    state_dict = replace_layer(state_dict, 'features.17', 'SPG_A_1.0')
    state_dict = replace_layer(state_dict, 'features.19', 'SPG_A_1.2')
    state_dict = replace_layer(state_dict, 'features.21', 'SPG_A_1.4')
    state_dict = replace_layer(state_dict, 'features.24', 'SPG_A_2.0')
    state_dict = replace_layer(state_dict, 'features.26', 'SPG_A_2.2')
    state_dict = replace_layer(state_dict, 'features.28', 'SPG_A_2.4')
    return state_dict


def load_pretrained_model(model, architecture_type, path=None):
    if path is not None:
        state_dict = torch.load(os.path.join(path, 'vgg16.pth'))
    else:
        state_dict = load_url(model_urls['vgg16'], progress=True)

    if architecture_type == 'spg':
        state_dict = batch_replace_layer(state_dict)
    state_dict = remove_layer(state_dict, 'classifier.')
    state_dict = adjust_pretrained_model(state_dict, model)

    model.load_state_dict(state_dict, strict=False)
    return model


def make_layers(cfg, **kwargs):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M1':
            layers += [nn.MaxPool2d(kernel_size=3, stride=2, padding=1)]
        elif v == 'M2':
            layers += [nn.MaxPool2d(kernel_size=3, stride=1, padding=1)]
        elif v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif v == 'A':
            layers += [
                ADL(kwargs['adl_drop_rate'], kwargs['adl_drop_threshold'])]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)

def _vgg(architecture_type, pretrained, pretrained_path, **kwargs):
    config_key = '28x28' if kwargs['large_feature_map'] else '14x14'
    layers = make_layers(configs_dict[architecture_type][config_key], **kwargs)
    unfreeze_layer = kwargs['unfreeze_layer']  

    model = {'cam': VggCam,
             'acol': VggAcol,
             'spg': VggSpg,
             'adl': VggCam}[architecture_type](layers, **kwargs)
    
    if kwargs['ft_ckpt'] is not None:
        print(f'[Debug][vgg.py] Load Fine-tuned Checkpoint: {kwargs["ft_ckpt"]}') 
        if unfreeze_layer in ('fc', 'conv'):
            print(f'[Debug][vgg.py] Load checkpoint to head2!') 
        else: print(f'[Debug][vgg.py] Load checkpoint to head1!')
        state_dict = torch.load(kwargs['ft_ckpt'])['state_dict'] 
        new_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('module.'): 
                new_key = key.replace('module.', '') 
            else:
                new_key = key
            if unfreeze_layer in ['fc', 'conv']:
                if new_key.startswith('fc.'):
                    new_key = new_key.replace('fc.', 'fc2.')
            new_state_dict[new_key] = value
        model.load_state_dict(new_state_dict, strict=False) 
    else:
        if pretrained:
            print(f'[Debug][vgg.py] Load pretrained model from {architecture_type}')
            model = load_pretrained_model(model, architecture_type,
                                      path=pretrained_path)
    return model

def vgg16(architecture_type, pretrained=False, pretrained_path=None,
          **kwargs):
    return _vgg(architecture_type, pretrained, pretrained_path, **kwargs)
