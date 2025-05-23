import torch
import numpy as np
from typing import List, Callable

import numpy as np
import cv2
import torch.nn.functional as F


class PerturbationConfidenceMetric:
    def __init__(self, perturbation):
        self.perturbation = perturbation

    def __call__(self, input_tensor: torch.Tensor, # batch_size x 3 x 224 x224
                 cams: np.ndarray, # batch_size x 224 x 224
                 targets: List[Callable],
                 model: torch.nn.Module,
                 return_visualization=False,
                 return_diff=True,
                 return_ori_scores=False):

        if return_diff:
            with torch.no_grad():
                outputs = model(input_tensor)
                if 'logits2' in outputs.keys():
                    outputs = outputs['logits2']
                else:
                    outputs = outputs['logits'] # batch_size x num_classes
                outputs = F.softmax(outputs, dim=1)
                scores = [target(output).cpu().numpy() for target, output in zip(targets, outputs)]
                scores = np.float32(scores)

        batch_size = input_tensor.size(0)
        perturbated_tensors = []
        for i in range(batch_size):
            cam = cams[i]
            tensor = self.perturbation(input_tensor[i, ...].cpu(),
                                       torch.from_numpy(cam))
            tensor = tensor.to(input_tensor.device) # 3 x 224 x 224
            perturbated_tensors.append(tensor.unsqueeze(0))
        perturbated_tensors = torch.cat(perturbated_tensors) # batch_size x 3 x 224 x 224

        with torch.no_grad():
            #Modified
            outputs = model(perturbated_tensors)
            if 'logits2' in outputs.keys():
                outputs_after_imputation = outputs['logits2']
            else:
                outputs_after_imputation = outputs['logits']
            outputs_after_imputation = F.softmax(outputs_after_imputation, dim=1)
        
        scores_after_imputation = [
            target(output).cpu().numpy() for target, output in zip(
                targets, outputs_after_imputation)]
        scores_after_imputation = np.float32(scores_after_imputation)

        if return_diff:
            result = scores_after_imputation - scores
        else:
            result = scores_after_imputation

        if return_visualization:
            if return_ori_scores:
                return result, perturbated_tensors, scores
            return result, perturbated_tensors
        else:
            if return_ori_scores:
                return result, scores
            return result


class RemoveMostRelevantFirst:
    def __init__(self, percentile, imputer):
        self.percentile = percentile
        self.imputer = imputer

    def __call__(self, input_tensor, mask):
        imputer = self.imputer
        if self.percentile != 'auto':
            threshold = np.percentile(mask.cpu().numpy(), self.percentile)
            binary_mask = np.float32(mask < threshold)
        else:
            _, binary_mask = cv2.threshold(
                np.uint8(mask * 255), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        binary_mask = torch.from_numpy(binary_mask)
        binary_mask = binary_mask.to(mask.device)
        return imputer(input_tensor, binary_mask)


class RemoveLeastRelevantFirst(RemoveMostRelevantFirst):
    def __init__(self, percentile, imputer):
        super(RemoveLeastRelevantFirst, self).__init__(percentile, imputer)

    def __call__(self, input_tensor, mask):
        return super(RemoveLeastRelevantFirst, self).__call__(
            input_tensor, 1 - mask)


class AveragerAcrossThresholds:
    def __init__(
        self,
        imputer,
        percentiles=[
            10,
            20,
            30,
            40,
            50,
            60,
            70,
            80,
            90]):
        self.imputer = imputer
        self.percentiles = percentiles

    def __call__(self,
                 input_tensor: torch.Tensor,
                 cams: np.ndarray,
                 targets: List[Callable],
                 model: torch.nn.Module):
        scores = []
        for percentile in self.percentiles:
            imputer = self.imputer(percentile)
            scores.append(imputer(input_tensor, cams, targets, model))
        return np.mean(np.float32(scores), axis=0)





class PerturbationConfidenceMetric_noDiff:
    def __init__(self, perturbation):
        self.perturbation = perturbation

    def __call__(self, input_tensor: torch.Tensor, # batch_size x 3 x 224 x224
                 cams: np.ndarray, # batch_size x 224 x 224
                 targets: List[Callable],
                 model: torch.nn.Module,
                 return_visualization=False):
        
        batch_size = input_tensor.size(0)
        reverse_cams = 1 - cams

        perturbated_tensors = [];
        reverse_perturbated_tensors = []; 

        for i in range(batch_size):
            tensor = self.perturbation(input_tensor[i, ...].cpu(), torch.from_numpy(cams[i]))
            tensor = tensor.to(input_tensor.device) # 3 x 224 x 224
            perturbated_tensors.append(tensor.unsqueeze(0))

            reverse_tensor = self.perturbation(input_tensor[i, ...].cpu(), torch.from_numpy(reverse_cams[i]))
            reverse_tensor = reverse_tensor.to(input_tensor.device) # 3 x 224 x 224
            reverse_perturbated_tensors.append(reverse_tensor.unsqueeze(0))

        perturbated_tensors = torch.cat(perturbated_tensors)
        reverse_perturbated_tensors = torch.cat(reverse_perturbated_tensors)

        with torch.no_grad():
            outputs = model(perturbated_tensors)
            if 'logits2' in outputs.keys():
                outputs_after_imputation = outputs['logits2']
            else:
                outputs_after_imputation = outputs['logits']
            outputs_after_imputation = F.softmax(outputs_after_imputation, dim=1)

            reverse_outputs = model(reverse_perturbated_tensors)
            if 'logits2' in reverse_outputs.keys():
                reverse_outputs_after_imputation = reverse_outputs['logits2']
            else:
                reverse_outputs_after_imputation = reverse_outputs['logits']
            reverse_outputs_after_imputation = F.softmax(reverse_outputs_after_imputation, dim=1) 
            
        scores_after_imputation = [
            target(output).cpu().numpy() \
                for target, output in zip(targets, outputs_after_imputation)]
        scores_after_imputation = np.float32(scores_after_imputation)

        reverse_scores_after_imputation = [
            target(output).cpu().numpy() \
                for target, output in zip(targets, reverse_outputs_after_imputation)]
        reverse_scores_after_imputation = np.float32(reverse_scores_after_imputation)

        if return_visualization:
            return scores_after_imputation, reverse_scores_after_imputation, perturbated_tensors
        else:
            return scores_after_imputation, reverse_scores_after_imputation


class PerturbationConfidenceMetricAll:
    def __init__(self, perturbation):
        self.perturbation = perturbation
        self.percentiles = [i for i in range(0, 101)] # each 1% ~ 100%

    def __call__(self, input_tensor: torch.Tensor, # batch_size x 3 x 224 x224
                 muted_cams: np.ndarray, 
                 thresholded_cams:np.array,
                 targets: List[Callable],
                 model: torch.nn.Module):
        
        with torch.no_grad():
            outputs = model(input_tensor)
            if 'logits2' in outputs.keys():
                outputs = outputs['logits2']
            else:
                outputs = outputs['logits']
            outputs = F.softmax(outputs, dim=1)
            ori_scores = [target(output).cpu().numpy() for target, output in zip(targets, outputs)]
            ori_scores = np.float32(ori_scores)
        

        batch_size = input_tensor.size(0)
        reverse_cams = 1 - thresholded_cams


        muted_perturbated_tensors= [];
        thrs_perturbated_tensors = [];
        thrs_reverse_perturbated_tensors = []; 

        for i in range(batch_size):
            muted_tensor = self.perturbation(input_tensor[i, ...].cpu(), torch.from_numpy(muted_cams[i]))
            muted_tensor = muted_tensor.to(input_tensor.device) # 3 x 224 x 224
            muted_perturbated_tensors.append(muted_tensor.unsqueeze(0))

            thrs_tensor = self.perturbation(input_tensor[i, ...].cpu(), torch.from_numpy(thresholded_cams[i]))
            thrs_tensor = thrs_tensor.to(input_tensor.device) # 3 x 224 x 224
            thrs_perturbated_tensors.append(thrs_tensor.unsqueeze(0))

            reverse_tensor = self.perturbation(input_tensor[i, ...].cpu(), torch.from_numpy(reverse_cams[i]))
            reverse_tensor = reverse_tensor.to(input_tensor.device) # 3 x 224 x 224
            thrs_reverse_perturbated_tensors.append(reverse_tensor.unsqueeze(0))

        muted_perturbated_tensors = torch.cat(muted_perturbated_tensors)
        thrs_perturbated_tensors = torch.cat(thrs_perturbated_tensors)
        thrs_reverse_perturbated_tensors = torch.cat(thrs_reverse_perturbated_tensors)

        with torch.no_grad():
            muted_outputs = model(muted_perturbated_tensors)
            if 'logits2' in muted_outputs.keys():
                muted_outputs_after_imputation = muted_outputs['logits2']
            else:
                muted_outputs_after_imputation = muted_outputs['logits']
            muted_outputs_after_imputation = F.softmax(muted_outputs_after_imputation, dim=1)


            thrs_outputs = model(thrs_perturbated_tensors)
            if 'logits2' in thrs_outputs.keys():
                thrs_outputs_after_imputation = thrs_outputs['logits2']
            else:
                thrs_outputs_after_imputation = thrs_outputs['logits']
            thrs_outputs_after_imputation = F.softmax(thrs_outputs_after_imputation, dim=1)

            thrs_reverse_outputs = model(thrs_reverse_perturbated_tensors)
            if 'logits2' in thrs_reverse_outputs.keys():
                thrs_reverse_outputs_after_imputation = thrs_reverse_outputs['logits2']
            else:
                thrs_reverse_outputs_after_imputation = thrs_reverse_outputs['logits']
            thrs_reverse_outputs_after_imputation = F.softmax(thrs_reverse_outputs_after_imputation, dim=1) 


        muted_scores_after_imputation = [
            target(output).cpu().numpy() \
                for target, output in zip(targets, muted_outputs_after_imputation)]
        muted_scores_after_imputation = np.float32(muted_scores_after_imputation)

        muted_diff = muted_scores_after_imputation - ori_scores

        thrs_scores_after_imputation = [
            target(output).cpu().numpy() \
                for target, output in zip(targets, thrs_outputs_after_imputation)]
        thrs_scores_after_imputation = np.float32(thrs_scores_after_imputation)

        thrs_reverse_scores_after_imputation = [
            target(output).cpu().numpy() \
                for target, output in zip(targets, thrs_reverse_outputs_after_imputation)]
        thrs_reverse_scores_after_imputation = np.float32(thrs_reverse_scores_after_imputation)

        avg_drop = (np.maximum(-muted_diff, 0) / ori_scores) 
        avg_increase = np.float32(muted_diff > 0)

        return avg_drop, avg_increase, \
            thrs_scores_after_imputation, thrs_reverse_scores_after_imputation

