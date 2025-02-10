import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import math

    
# https://github.com/vladimirstarygin/Subcenter-ArcFace-Pytorch/blob/main/data_filtering/src/train_utils/losses/Subcenter_arcface.py
class SubcenterArcMarginProduct(nn.Module):
    def __init__(self, K=3, s=30.0, m=0.50, easy_margin=False):
        super(SubcenterArcMarginProduct, self).__init__()
        self.K = K
        self.s = s
        self.m = m

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def scale(self, logits):
        n_classess = logits.size(1)
        
        if self.s == "auto":
            return logits * math.sqrt(2) * math.log(n_classess - 1)
        else:
           return logits * self.s

    def forward(self, cosine, label):
        # Determine the number of classes
        n_classess = cosine.size(1) // self.K
        
        # Aggregate from K subcenters
        if self.K > 1:
            cosine = torch.reshape(cosine, (-1, n_classess, self.K))
            cosine, _ = torch.max(cosine, axis=2)
        
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        # Calculate phi - the angle between the logits and the target class
        phi = cosine * self.cos_m - sine * self.sin_m

        # Apply the margin
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # Convert label to one-hot
        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)
        
        # Calculate the output logits
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        # Scale output
        output = self.scale(output)

        return output
    
# https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py
class ArcMarginProduct(nn.Module):
    def __init__(self, s=30.0, m=0.50, easy_margin=False):
        super(ArcMarginProduct, self).__init__()
        self.s = s
        self.m = m

        self.easy_margin = easy_margin
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m
        
    def scale(self, logits):
        n_classess = logits.size(1)
        
        if self.s == "auto":
            return logits * math.sqrt(2) * math.log(n_classess - 1)
        else:
           return logits * self.s

    def forward(self, cosine, label):
        sine = torch.sqrt((1.0 - torch.pow(cosine, 2)).clamp(0, 1))
        
        phi = cosine * self.cos_m - sine * self.sin_m
        
        if self.easy_margin:
            phi = torch.where(cosine > 0, phi, cosine)
        else:
            phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        one_hot = torch.zeros(cosine.size(), device=cosine.device)
        one_hot.scatter_(1, label.view(-1, 1).long(), 1)

        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        
        output = self.scale(output)

        return output
    
# https://github.com/KaiyangZhou/pytorch-center-loss/blob/master/center_loss.py
class CenterLoss(nn.Module):
    def __init__(self, num_classes=10, feat_dim=2, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = Parameter(torch.randn(self.num_classes, self.feat_dim).cuda())
        else:
            self.centers = Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat = distmat.addmm(x, self.centers.t(), alpha=-2, beta=1)

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu: classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-9, max=1e+9).sum() / batch_size

        return loss
    
class FocalLoss(nn.Module):
    """
    Classic Focal Loss for multi-class classification
    
    Args:
        gamma: focusing parameter
        alpha: can be a float or a list/array for class-wise weighting
    """
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        super().__init__()
        if isinstance(alpha, (float, int)):
            # same alpha for all classes
            self.alpha = alpha
        else:
            # if alpha is a list/array, convert to tensor
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        """
        Args:
            inputs: (N, C), raw logits from network output
            targets: (N,) or (N, 1), ground truth labels
        """
        log_probs = F.log_softmax(inputs, dim=-1) # (N, C)
        probs = torch.exp(log_probs) # (N, C)
        
        # Gather log_probs for the correct class
        focal_part = (1 - probs) ** self.gamma # (N, C)
        
        # If alpha is per-class, gather for targets
        if isinstance(self.alpha, torch.Tensor):
            alpha_factor = self.alpha[targets]
        else:
            alpha_factor = self.alpha
        
        # NLL
        loss = -alpha_factor * focal_part * log_probs

        # Pick the loss values corresponding to the correct class
        loss = loss.gather(dim=-1, index=targets.unsqueeze(-1))
        loss = loss.squeeze(-1) # (N,)

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss

    
# https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py
# class AddMarginProduct(nn.Module):
#     r"""Implement of large margin cosine distance: :
#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         s: norm of input feature
#         m: margin
#         cos(theta) - m
#     """

#     def __init__(self, in_features, out_features, s=30.0, m=0.40):
#         super(AddMarginProduct, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.s = s
#         self.m = m
#         self.weight = Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)

#     def forward(self, input, label):
#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         cosine = F.linear(F.normalize(input), F.normalize(self.weight))
#         phi = cosine - self.m
#         # --------------------------- convert label to one-hot ---------------------------
#         one_hot = torch.zeros(cosine.size(), device=input.device)
#         # one_hot = one_hot.cuda() if cosine.is_cuda else one_hot
#         one_hot.scatter_(1, label.view(-1, 1).long(), 1)
#         # -------------torch.where(out_i = {x_i if condition_i else y_i) -------------
#         output = (one_hot * phi) + ((1.0 - one_hot) * cosine)  # you can use torch.where if your torch.__version__ is 0.4
#         output *= self.s

#         return output
   
# https://github.com/ronghuaiyang/arcface-pytorch/blob/47ace80b128042cd8d2efd408f55c5a3e156b032/models/metrics.py 
# class SphereProduct(nn.Module):
#     r"""Implement of large margin cosine distance: :
#     Args:
#         in_features: size of each input sample
#         out_features: size of each output sample
#         m: margin
#         cos(m*theta)
#     """
#     def __init__(self, in_features, out_features, m=4):
#         super(SphereProduct, self).__init__()
#         self.in_features = in_features
#         self.out_features = out_features
#         self.m = m
#         self.base = 1000.0
#         self.gamma = 0.12
#         self.power = 1
#         self.LambdaMin = 5.0
#         self.iter = 0
#         self.weight = Parameter(torch.FloatTensor(out_features, in_features))
#         nn.init.xavier_uniform_(self.weight)

#         # duplication formula
#         self.mlambda = [
#             lambda x: x ** 0,
#             lambda x: x ** 1,
#             lambda x: 2 * x ** 2 - 1,
#             lambda x: 4 * x ** 3 - 3 * x,
#             lambda x: 8 * x ** 4 - 8 * x ** 2 + 1,
#             lambda x: 16 * x ** 5 - 20 * x ** 3 + 5 * x
#         ]

#     def forward(self, input, label):
#         # lambda = max(lambda_min,base*(1+gamma*iteration)^(-power))
#         self.iter += 1
#         self.lamb = max(self.LambdaMin, self.base * (1 + self.gamma * self.iter) ** (-1 * self.power))

#         # --------------------------- cos(theta) & phi(theta) ---------------------------
#         cos_theta = F.linear(F.normalize(input), F.normalize(self.weight))
#         cos_theta = cos_theta.clamp(-1, 1)
#         cos_m_theta = self.mlambda[self.m](cos_theta)
#         theta = cos_theta.data.acos()
#         k = (self.m * theta / 3.14159265).floor()
#         phi_theta = ((-1.0) ** k) * cos_m_theta - 2 * k
#         NormOfFeature = torch.norm(input, 2, 1)

#         # --------------------------- convert label to one-hot ---------------------------
#         one_hot = torch.zeros(cos_theta.size(), device=input.device)
#         one_hot.scatter_(1, label.view(-1, 1), 1)

#         # --------------------------- Calculate output ---------------------------
#         output = (one_hot * (phi_theta - cos_theta) / (1 + self.lamb)) + cos_theta
#         output *= NormOfFeature.view(-1, 1)

#         return output