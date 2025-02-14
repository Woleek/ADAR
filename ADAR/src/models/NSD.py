# Adapted from https://github.com/piotrkawa/audio-deepfake-source-tracing

from abc import ABC, abstractmethod
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F


class OODDetector(ABC):
    @abstractmethod
    def setup(self, args, train_model_outputs: Dict[str, torch.Tensor]):
        pass

    @abstractmethod
    def infer(self, model_outputs: Dict[str, torch.Tensor]) -> torch.Tensor:
        pass


def NSD_with_angle(feats_train, feats, min=False):
    feas_train = feats_train.cpu().numpy()
    feats = feats.cpu().numpy()
    cos_similarity = np.dot(feats, feats_train.T)
    if min:
        scores = np.array(cos_similarity.min(axis=1))
    else:
        scores = np.array(cos_similarity.mean(axis=1))
    return scores


class NSDOODDetector(OODDetector):
    def setup(self, train_model_outputs):
        # Compute the training set info
        logits_train = torch.Tensor(train_model_outputs["logits"])
        feats_train = torch.Tensor(train_model_outputs["feats"])
        train_labels = train_model_outputs["labels"]
        
        feats_train = F.normalize(feats_train, p=2, dim=-1)
        confs_train = torch.logsumexp(logits_train, dim=1)
        
        self.scaled_feats_train = feats_train * confs_train[:, None]

    def infer(self, model_outputs):
        feats = torch.Tensor(model_outputs["feats"])
        logits = torch.Tensor(model_outputs["logits"])
        
        feats = F.normalize(feats, p=2, dim=-1)
        confs = torch.logsumexp(logits, dim=1)
        guidances = NSD_with_angle(self.scaled_feats_train, feats)
        scores = torch.from_numpy(guidances).to(confs.device) * confs
        return scores
        
        
class HierNSDOODDetector(OODDetector):
    def __init__(self, hierarchy_type, confidence_scaling='local'):
        """
        confidence_scaling: 'local' | 'sup' | 'avg' | 'none'
        """
        self.hierarchy_type = hierarchy_type
        self.confidence_scaling = confidence_scaling

    def setup(self, train_model_outputs):
        feats_train = torch.Tensor(train_model_outputs["feats"])
        sup_logits_train = torch.Tensor(train_model_outputs["sup_logits"])
        logits_train = torch.Tensor(train_model_outputs["logits"])  # global (LCL) or local (LCPN)
        sup_train_labels = train_model_outputs["sup_labels"]
        train_labels = train_model_outputs["labels"]

        # Normalize features
        feats_train = F.normalize(feats_train, p=2, dim=-1)

        # Compute confidence (energy) scores
        sup_confs_train = torch.logsumexp(sup_logits_train, dim=1)
        confs_train = torch.logsumexp(logits_train, dim=1)

        # Compute scaled features for NSD
        self.sup_scaled_feats_train = feats_train * sup_confs_train[:, None]
        self.scaled_feats_train = feats_train * confs_train[:, None]

        # For LCPN, create a per-superclass pool
        if self.hierarchy_type == 'LCPN':
            self.scaled_feats_per_sup = {}
            unique_sups = sorted(set(sup_train_labels))
            for sup in unique_sups:
                # mask to get only training samples with this sup label
                mask = (sup_train_labels == sup)
                self.scaled_feats_per_sup[sup] = feats_train[mask] * sup_confs_train[mask, None]

    def infer(self, model_outputs):
        sup_logits = torch.Tensor(model_outputs["sup_logits"])
        logits = torch.Tensor(model_outputs["logits"])
        feats = torch.Tensor(model_outputs["feats"])

        feats = F.normalize(feats, p=2, dim=-1)
        device = feats.device

        sup_confs = torch.logsumexp(sup_logits, dim=1)
        local_confs = torch.logsumexp(logits, dim=1)

        # Global NSD score at the superclass level
        sup_guidances = NSD_with_angle(self.sup_scaled_feats_train, feats)
        sup_scores = torch.from_numpy(sup_guidances).to(device) * sup_confs

        if self.hierarchy_type == 'LCPN':
            sup_preds = torch.argmax(sup_logits, dim=1)
            scores = torch.zeros(feats.size(0), device=device)
            
            for sup in torch.unique(sup_preds):
                # Dot product with that supclass training pool
                mask = (sup_preds == sup)
                sup_pool = self.scaled_feats_per_sup[sup.item()]
                guidances = NSD_with_angle(sup_pool, feats[mask])
                guidances = torch.from_numpy(guidances).to(device)
                
                if self.confidence_scaling == 'local':
                    # Measure how in-distribution of a local classifier the sample is
                    scaling = local_confs[mask]
                    
                elif self.confidence_scaling == 'sup':
                    # Measure how in-distribution of a superclass the sample is
                    scaling = sup_confs[mask]
                    
                elif self.confidence_scaling == 'avg':
                    # Average of local and superclass confidences
                    scaling = (local_confs[mask] + sup_confs[mask]) / 2.0
                    
                elif self.confidence_scaling == 'none':
                    # No confidence scaling
                    scaling = torch.ones_like(local_confs[mask])
                    
                else:
                    raise ValueError(f"Unknown confidence scaling type: {self.confidence_scaling}")
                
                partial_scores = guidances * scaling
                scores[mask] = partial_scores
        else:
            # For non-hierarchical (e.g., LCL), use global NSD score
            guidances = NSD_with_angle(self.scaled_feats_train, feats)
            scores = torch.from_numpy(guidances).to(device) * local_confs

        # Return tuple (sup_scores, final_scores)
        return (sup_scores, scores)
