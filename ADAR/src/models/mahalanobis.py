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
    
def mahalanobis_distance(feats, inv_cov, means_dict):
    distances = []
    for label, mean in means_dict.items():
        diff = feats - mean.unsqueeze(0)
        # Mahalanobis distance: sqrt(diff^T * inv_cov * diff)
        d = torch.sqrt((diff @ inv_cov * diff).sum(dim=1))
        distances.append(d)
    distances = torch.stack(distances, dim=1) # shape: [N, num_classes]
    
    # Return the minimum distance to any class
    min_distances, _ = distances.min(dim=1)
    return min_distances

class MahalanobisOODDetector(OODDetector):
    def setup(self, train_model_outputs):
        # Get training features and global labels
        feats_train = torch.Tensor(train_model_outputs["feats"])
        train_labels = train_model_outputs["labels"]
        
        # Normalize features
        feats_train = F.normalize(feats_train, p=2, dim=-1)
        self.train_feats = feats_train  # normalized features
        self.train_labels = np.array(train_labels)
        
        # Compute pooled covariance with regularization
        X = feats_train.cpu().numpy()  # shape [N, D]
        cov = np.cov(X, rowvar=False)
        reg = 1e-6 * np.eye(cov.shape[0])
        cov += reg
        inv_cov = np.linalg.inv(cov)
        self.inv_cov = torch.tensor(inv_cov, dtype=torch.float32, device=feats_train.device)
        
        # Compute per-class means
        unique_labels = np.unique(self.train_labels)
        self.class_means = {}
        for label in unique_labels:
            mask = (np.array(train_labels) == label)
            class_feats = feats_train[mask]
            self.class_means[label] = class_feats.mean(dim=0)
    
    def infer(self, model_outputs):
        # Get features and normalize
        feats = torch.Tensor(model_outputs["feats"])
        feats = F.normalize(feats, p=2, dim=-1)
        
        # Compute Mahalanobis scores
        scores = mahalanobis_distance(feats, self.inv_cov, self.class_means)
        return scores
    
class HierMahalanobisOODDetector(OODDetector):
    def __init__(self, hierarchy_type):
        self.hierarchy_type = hierarchy_type

    def setup(self, train_model_outputs):
        # Get training features and global labels
        feats_train = torch.Tensor(train_model_outputs["feats"])
        train_labels = train_model_outputs["labels"]  # global labels
        
        # Normalize features
        feats_train = F.normalize(feats_train, p=2, dim=-1)
        self.train_feats = feats_train  # normalized features
        self.train_labels = np.array(train_labels)
        
        # Compute pooled covariance (global) with regularization
        X = feats_train.cpu().numpy()  # shape [N, D]
        cov = np.cov(X, rowvar=False)
        reg = 1e-6 * np.eye(cov.shape[0])
        cov += reg
        inv_cov = np.linalg.inv(cov)
        self.inv_cov = torch.tensor(inv_cov, dtype=torch.float32, device=feats_train.device)
        
        # Compute per-class (global) means
        unique_labels = np.unique(self.train_labels)
        self.class_means = {}
        for label in unique_labels:
            mask = (np.array(train_labels) == label)
            class_feats = feats_train[mask]
            self.class_means[label] = class_feats.mean(dim=0)
        
        # Compute per-superclass means from training outputs
        sup_train_labels = np.array(train_model_outputs["sup_labels"])
        unique_sup = np.unique(sup_train_labels)
        self.sup_means = {}
        for sup in unique_sup:
            mask = (sup_train_labels == sup)
            sup_feats = feats_train[mask]
            self.sup_means[sup] = sup_feats.mean(dim=0)

    def infer(self, model_outputs):
        # Get features and normalize
        feats = torch.Tensor(model_outputs["feats"])
        feats = F.normalize(feats, p=2, dim=-1)

        # Compute global Mahalanobis scores using global class means
        global_scores = mahalanobis_distance(feats, self.inv_cov, self.class_means)

        # Compute superclass scores:
        sup_logits = torch.Tensor(model_outputs["sup_logits"])
        sup_preds = torch.argmax(sup_logits, dim=1)
        sup_scores = torch.zeros(feats.size(0), device=feats.device)
        
        # For each predicted superclass, compute Mahalanobis distance relative to its mean
        for sup in torch.unique(sup_preds):
            mask = (sup_preds == sup)
            sup_scores[mask] = mahalanobis_distance(feats[mask], self.inv_cov, {sup: self.sup_means[sup.item()]})

        return (sup_scores, global_scores)