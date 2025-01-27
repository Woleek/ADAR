# Adapted from https://github.com/piotrkawa/audio-deepfake-source-tracing

import argparse
import json
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data.sampler as torch_sampler
from torch.utils.data import DataLoader
from tqdm import tqdm

from src import utils
from src.datasets.utils import HuggingFaceFeatureExtractor
from src.datasets.dataset import MLAADFD_AR_Dataset, MLAADFDDataset
from src.models.w2v2_aasist import W2VAASIST_HMulT#, W2VAASIST_HCas
from src.lossess import ArcMarginProduct, SubcenterArcMarginProduct, CenterLoss


def parse_args():
    parser = argparse.ArgumentParser("Training script parameters")

    # Paths to features and output
    parser.add_argument(
        "-f",
        "--path_to_dataset",
        type=str,
        default="data/prepared_ds/",
        help="Path to the previuosly extracted features",
    )
    parser.add_argument(
        "--is_segmented",
        type=bool,
        default=True,
        help="If audio samples in dataset are split into segments",
    )
    parser.add_argument(
        "--out_folder", type=str, default="exp/trained_models/base", help="Output folder"
    )
    
    # Augmentation parameters
    parser.add_argument(
        "--pre_augmented",
        type=bool,
        default=False,
        help="If the dataset is already augmented (turns off augmentation)",
    )
    parser.add_argument(
        "--musan_path",
        type=str,
        default="data/musan/",
        help="Path to the MUSAN dataset",
    )
    parser.add_argument(
        "--rir_path",
        type=str,
        default="data/rirs/",
        help="Path to RIRs dataset",
    )
    parser.add_argument(
        "--sampling_rate", type=int, default=16_000, help="Audio sampling rate"
    )
    
    # HuggingFace feature extractor
    parser.add_argument(
        "--pre_encoded",
        type=bool,
        default=False,
        help="If the dataset is already encoded (turns off feature extraction)",
    )
    parser.add_argument(
        "--model_class",
        type=str,
        default="Wav2Vec2Model",
        help="Class of the feature extractor",
    )
    parser.add_argument(
        "--model_layer",
        type=int,
        default=5,
        help="Which layer to use from the feature extractor",
    )
    parser.add_argument(
        "--hugging_face_path",
        type=str,
        default="facebook/wav2vec2-base",
        help="Path from the HF collections",
    )
    
    # Hierarchical classification
    parser.add_argument(
        "--hierarchy_type",
        type=str,
        default="multi-task",
        choices=["multi-task", "cascade"],
        help="Type of training for hierarchical classification (multi-task or cascade)",
    )
    parser.add_argument(
        "--superclass_lut",
        type=str,
        default="superclass_mapping_known.csv",
        help="File with superclass mapping for hierarchical classification",
    )

    # Training hyperparameters
    parser.add_argument("--seed", type=int, help="random number seed", default=688)
    parser.add_argument(
        "--feat_dim",
        type=int,
        default=768,
        help="Feature dimension from the wav2vec model",
    )
    parser.add_argument(
        "--num_classes", type=int, default=24, help="Number of in domain classes"
    )
    parser.add_argument(
        "--num_epochs", type=int, default=30, help="Number of epochs for training"
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--weighted_sampling", type=bool, default=False, help="Draw samples from train dataset with weighted probability based on class distribution"
    )
    parser.add_argument("--lr", type=float, default=0.0005, help="learning rate")
    parser.add_argument(
        "--lr_decay", type=float, default=0.5, help="decay learning rate"
    )
    parser.add_argument("--interval", type=int, default=10, help="interval to decay lr")
    parser.add_argument("--beta_1", type=float, default=0.9, help="bata_1 for Adam")
    parser.add_argument("--beta_2", type=float, default=0.999, help="beta_2 for Adam")
    parser.add_argument("--eps", type=float, default=1e-8, help="epsilon for Adam")
    parser.add_argument("--num_workers", type=int, default=0, help="number of workers")
    parser.add_argument(
        "--base_loss",
        type=str,
        default="ce",
        choices=["ce"],
        help="Loss for basic training",
    )
    parser.add_argument(
        "--label_smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor [0.0 - 1.0] for the loss function"
    )
    
    # Resume training
    parser.add_argument(
        "--resume-checkpoint",
        type=str,
        help="Resume training from given model checkpoint",
        default=None,
    )
    parser.add_argument(
        "--resume-epoch",
        type=int,
        help="Resume training from given epoch number",
        default=None,
    )
    parser.add_argument(
        "--resume-optimizer",
        type=str,
        help="Resume training from given optimizer state",
        default=None,
    )
    
    # Additional loss functions
    def float_or_str(value):
        try:
            return float(value)
        except ValueError:
            return value
    
    parser.add_argument("--use_arc_margin", action="store_true", help="Use ArcMarginProduct")
    parser.add_argument("--use_sub-center-arc_margin", action="store_true", help="Use Sub-Center ArcMarginProduct")
    
    parser.add_argument("--arc_s", type=float_or_str, default="auto", help="Scale parameter s for ArcMarginProduct")
    parser.add_argument("--arc_m", type=float, default=0.5, help="Margin parameter m for ArcMarginProduct")
    parser.add_argument("--easy_margin", type=bool, default=False, help="Use easy margin in ArcMarginProduct")
    parser.add_argument("--optimize_arc_margin_weights", type=bool, default=True, help="Optimize ArcMarginProduct weights")
    parser.add_argument("--k_centers", type=int, default=1, help="Number of centers for Sub-Center ArcMarginProduct")
    
    parser.add_argument("--use_center_loss", action="store_true", help="Use CenterLoss")
    parser.add_argument("--center_loss_weight", type=float, default=0.01, help="Weight for CenterLoss")
    parser.add_argument("--resume_center_loss_optimizer", type=str, help="Resume training from given center loss optimizer state", default=None)
    
    args = parser.parse_args()

    # Set seeds
    utils.set_seed(args.seed)

    # Path for output data
    if not os.path.exists(args.out_folder):
        os.makedirs(args.out_folder)

    # Folder for intermediate results
    if not os.path.exists(os.path.join(args.out_folder, "checkpoint")):
        os.makedirs(os.path.join(args.out_folder, "checkpoint"))

    # Path for input data
    assert os.path.exists(args.path_to_dataset)

    # Save training arguments
    with open(os.path.join(args.out_folder, "args.json"), "w") as file:
        file.write(json.dumps(vars(args), sort_keys=True, separators=(",\n", ":")))

    cuda = torch.cuda.is_available()
    print("Running on: ", "cuda" if cuda else "cpu")
    args.device = torch.device("cuda" if cuda else "cpu")
    return args


def train(args):
    # Load superclass LUT
    id_map, _ = utils.load_superclass_mapping(args.path_to_dataset, args.superclass_lut)
        
    # Load the train and dev data (only known classes)
    print("Loading training data...")
    if not args.pre_encoded:
        training_set = MLAADFD_AR_Dataset(args.path_to_dataset, args.pre_augmented, args.sampling_rate, args.musan_path, args.rir_path, "train", segmented=args.is_segmented, superclass_mapping=id_map)
    else:
        training_set = MLAADFDDataset(args.path_to_dataset, "train", superclass_mapping=id_map)
        
    print("\nLoading dev data...")
    if not args.pre_encoded:
        dev_set = MLAADFD_AR_Dataset(args.path_to_dataset, args.pre_augmented, args.sampling_rate, args.musan_path, args.rir_path, "dev", mode="known", segmented=args.is_segmented, superclass_mapping=id_map)
    else:
        dev_set = MLAADFDDataset(args.path_to_dataset, "dev", mode="known", superclass_mapping=id_map)
    
    if args.weighted_sampling:
        # TODO: Implement hierarchy weighting as well
        train_sampler = torch_sampler.WeightedRandomSampler(
            training_set.sample_weights, len(training_set), 
            replacement=False # No oversampling
        )
        print(f"Using weighted sampling (undersampling)")
    else:
        train_sampler = torch_sampler.SubsetRandomSampler(range(len(training_set)))

    train_loader = DataLoader(
        training_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=train_sampler,
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=torch_sampler.SubsetRandomSampler(range(len(dev_set))),
    )
    
    start_epoch = 0
    
    # Set up loss functions
    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss(
            label_smoothing=args.label_smoothing
        )
    else:
        raise ValueError(f"Loss function {args.base_loss} not supported")
    
    if args.use_arc_margin and args.use_sub_center_arc_margin:
        raise ValueError("Cannot use both ArcMarginProduct and SubcenterArcMarginProduct at the same time")
        
    if args.use_arc_margin:
        print("[INFO] Using ArcFace Loss...")
        arc_margin = ArcMarginProduct(
            s=args.arc_s,
            m=args.arc_m,
            easy_margin=args.easy_margin
        ).to(args.device)
        
    if args.use_sub_center_arc_margin:
        print(f"[INFO] Using Sub-Center ArcFace Loss with {args.k_centers} centers per class...")
        arc_margin = SubcenterArcMarginProduct(
            K=args.k_centers,
            s=args.arc_s,
            m=args.arc_m,
            easy_margin=args.easy_margin
        ).to(args.device)
        
    if args.use_center_loss:
        print("[INFO] Using Center Loss...")
        center_loss_fn = CenterLoss(
            num_classes=args.num_classes,
            feat_dim=5 * 32, # Last hidden output size of W2VAASIST
            use_gpu=(args.device.type == "cuda")
        )
        center_loss_optimizer = torch.optim.SGD(center_loss_fn.parameters(), lr=0.5) # Separate optimizer for center-loss parameters
        
    # Set up feature extractor
    if not args.pre_encoded:
        feature_extractor = HuggingFaceFeatureExtractor(
            model_class_name=args.model_class,
            layer=args.model_layer,
            name=args.hugging_face_path
        )
        
        # Freeze the feature extractor
        for param in feature_extractor.model.parameters():
            param.requires_grad = False
            
        feature_extractor.model.eval()
        
    # Setup the model to learn in-domain classess
    if args.use_sub_center_arc_margin:
        raise ValueError("Not implemented with Subcenter yet")
    
    if args.hierarchy_type == "multi-task":
        num_sup_classes = len(set(id_map.values())) # Number of superclasses
        num_sub_classes = args.num_classes # Number of original (global) classes
        
        model = W2VAASIST_HMulT(
            feature_dim=args.feat_dim, 
            num_suplabels=num_sup_classes, 
            num_labels=num_sub_classes, 
            normalize_before_output=True # ArcMargin expects normalized embeddings
                if (args.use_arc_margin or args.use_sub_center_arc_margin) 
                else False
        ).to(args.device)
        
    elif args.hierarchy_type == "cascade":
        raise ValueError("Not implemented with cascade yet")
    
    else:
        raise ValueError(f"Hierarchy type {args.hierarchy_type} not supported")
    
    if args.resume_checkpoint and args.resume_epoch:
        print(f"Resuming training from checkpoint {args.resume_checkpoint} at epoch {args.resume_epoch}")
        model.load_state_dict(torch.load(args.resume_checkpoint, weights_only=True))
        
        start_epoch = int(args.resume_epoch)
        
    # Main optimizer + arc_margin params (optional)
    feat_optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta_1, args.beta_2),
        eps=args.eps,
        weight_decay=0.0005
    )
    
    if args.resume_optimizer:
        feat_optimizer.load_state_dict(torch.load(args.resume_optimizer, weights_only=True))
        
    if args.resume_center_loss_optimizer:
        center_loss_optimizer.load_state_dict(torch.load(args.resume_center_loss_optimizer, weights_only=True))
    
    print(f"Training a {type(model).__name__} model for {args.num_epochs} epochs")

    prev_loss = 1e8
    # Main training loop
    for epoch_num in range(start_epoch, args.num_epochs + start_epoch):
        model.train()
        utils.adjust_learning_rate(args, args.lr, feat_optimizer, epoch_num)

        epoch_bar = tqdm(train_loader, desc=f"Epoch [{epoch_num+1}/{args.num_epochs + start_epoch}]")
        
        train_sup_accuracy, train_sub_accuracy, train_loss = [], [], [] 
        for iter_num, batch in enumerate(epoch_bar):
            feat, _, labels = batch # audio_sample, path, class_id
            
            sup_labels, sub_labels = labels
            sup_labels = sup_labels.to(args.device)
            sub_labels = sub_labels.to(args.device)
            
            if not args.pre_encoded: # Extract features
                if args.is_segmented:
                    n_segments = feat.shape[1]
                    feat = feat.view(-1, args.sampling_rate) # [batch_size * num_segments, sampling_rate]
                    
                with torch.no_grad():
                    feat = feature_extractor(feat, args.sampling_rate).float()
                
                if args.is_segmented:
                    feat = feat.view(-1, n_segments, *feat.shape[1:]) # [batch_size, num_segments, *embed_dims]
                    feat = feat.mean(dim=1) # [batch_size, *embed_dims] average over segments
                    
            feat = feat.transpose(1, 2).to(args.device)

            # ---- Forward pass ----
            feats, logits = model(feat) # feats - last hidden, logits - model output from linear(s)
            sup_logits, sub_logits = logits
            
            total_loss = 0
            if args.use_arc_margin or args.use_sub_center_arc_margin: # Apply ArcMarginProduct
                sup_logits = arc_margin(sup_logits, sup_labels) 
                sub_logits = arc_margin(sub_logits, sub_labels)
                
                loss_sup = criterion(sup_logits, sup_labels)
                loss_sub = criterion(sub_logits, sub_labels)
                
                loss_base = 0.5 * (loss_sup + loss_sub) 
                
            else:
                loss_sup = criterion(sup_logits, sup_labels)
                loss_sub = criterion(sub_logits, sub_labels)
                
            loss_base = 0.5 * (loss_sup + loss_sub) # TODO: Consider weighting         
            total_loss += loss_base
            
            if args.use_center_loss:
                loss_center_sup = center_loss_fn(feats, sup_labels)
                loss_center_sub = center_loss_fn(feats, sub_labels)
                loss_center = 0.5 * (loss_center_sup + loss_center_sub)
                
                total_loss += args.center_loss_weight * loss_center

            # ---- Backprop ----
            feat_optimizer.zero_grad()
            if args.use_center_loss:
                center_loss_optimizer.zero_grad()
                
            total_loss.backward()
            
            feat_optimizer.step()
            if args.use_center_loss:
                # scale down center loss grads
                for param in center_loss_fn.parameters():
                    param.grad.data *= (1. / args.center_loss_weight)
                center_loss_optimizer.step()
            
            # ---- Scores ----
            with torch.no_grad():
                sup_score = F.softmax(sup_logits, dim=1)
                sub_score = F.softmax(sub_logits, dim=1)
                
                sup_predicted = torch.argmax(sup_score, dim=1)
                sub_predicted = torch.argmax(sub_score, dim=1)
                
                sup_acc = (sup_predicted == sup_labels).float().mean()
                sub_acc = (sub_predicted == sub_labels).float().mean()
                
            train_sup_accuracy.append(sup_acc.item())
            train_sub_accuracy.append(sub_acc.item())
            train_loss.append(total_loss.item())
            
            epoch_bar.set_postfix({
                "sup_acc": f"{sum(train_sup_accuracy)/(iter_num+1):.2f}",
                "sub_acc": f"{sum(train_sub_accuracy)/(iter_num+1):.2f}",
                "train_loss": f"{sum(train_loss)/(iter_num+1):.4f}"
            })
                
                
        epoch_train_loss = sum(train_loss) / (iter_num + 1)
        epoch_train_sup_acc = sum(train_sup_accuracy) / (iter_num + 1)
        epoch_train_sub_acc = sum(train_sub_accuracy) / (iter_num + 1)

        # Epoch eval
        model.eval()
        with torch.no_grad():
            val_bar = tqdm(dev_loader, desc=f"Validation for epoch {epoch_num+1}")
            val_sup_accuracy, val_sub_accuracy, val_loss = [], [], []
            for iter_num, batch in enumerate(val_bar):
                feat, _, labels = batch
                
                sup_labels, sub_labels = labels
                sup_labels = sup_labels.to(args.device)
                sub_labels = sub_labels.to(args.device)
                
                if not args.pre_encoded: # Extract features
                    if args.is_segmented:
                        n_segments = feat.shape[1]
                        feat = feat.view(-1, args.sampling_rate)
                    
                    feat = feature_extractor(feat, args.sampling_rate).float()
                    
                    if args.is_segmented:
                        feat = feat.view(-1, n_segments, *feat.shape[1:])
                        feat = feat.mean(dim=1)
                        
                feat = feat.transpose(1, 2).to(args.device)

                feats, logits = model(feat)
                sup_logits, sub_logits = logits
                
                # Use model's default output logits - do not apply margin
                if args.use_sub_center_arc_margin:
                    # Aggregate sub-center outputs
                    if arc_margin.K > 1:
                        sup_logits = torch.reshape(sup_logits, (-1, num_sup_classes, arc_margin.K))
                        sub_logits = torch.reshape(sub_logits, (-1, num_sub_classes, arc_margin.K))
                        
                        sup_logits, _ = torch.max(sup_logits, axis=2)
                        sub_logits, _ = torch.max(sub_logits, axis=2)
                        
                        sup_logits = arc_margin.scale(sup_logits) # TODO: Check if this is correct
                        sub_logits = arc_margin.scale(sub_logits)
                        
                elif args.use_arc_margin:
                    sup_logits = arc_margin.scale(sup_logits) # TODO: Check if this is correct
                    sub_logits = arc_margin.scale(sub_logits)
                
                sup_loss = criterion(sup_logits, sup_labels)
                sub_loss = criterion(sub_logits, sub_labels)
                loss = 0.5 * (sup_loss + sub_loss)
                
                sup_score = F.softmax(sup_logits, dim=1)
                sub_score = F.softmax(sub_logits, dim=1)
                
                sup_predicted = torch.argmax(sup_score, dim=1)
                sub_predicted = torch.argmax(sub_score, dim=1)
                
                sup_acc = (sup_predicted == sup_labels).float().mean()
                sub_acc = (sub_predicted == sub_labels).float().mean()
                
                val_sup_accuracy.append(sup_acc.item())
                val_sub_accuracy.append(sub_acc.item())
                val_loss.append(loss.item())
                
                val_bar.set_postfix(
                    {
                        "val_loss": f"{sum(val_loss)/(iter_num+1):.4f}",
                        "val_sup_acc": f"{sum(val_sup_accuracy)/(iter_num+1):.2f}",
                        "val_sub_acc": f"{sum(val_sub_accuracy)/(iter_num+1):.2f}",
                    }
                )

        epoch_val_loss = sum(val_loss) / (iter_num + 1)
        epoch_val_sup_acc = sum(val_sup_accuracy) / (iter_num + 1)
        epoch_val_sub_acc = sum(val_sub_accuracy) / (iter_num + 1)
        
        if epoch_val_loss < prev_loss:
            prev_loss = epoch_val_loss
            
            # Gather model state
            state = model.state_dict()
            if args.use_arc_margin or args.use_sub_center_arc_margin:         
                # Keep scale for inference
                state["arc_margin_scale"] = arc_margin.s
            if args.use_sub_center_arc_margin:
                # Keep K_centers for inference
                state["arc_margin_k_centers"] = arc_margin.K
            
            # Save the checkpoint with better val_loss
            utils.save_checkpoint(
                save_folder=args.out_folder,
                model_state=state,
                optimizer_state=feat_optimizer.state_dict(),
                center_loss_optimizer_state=center_loss_optimizer.state_dict() if args.use_center_loss else None,
                training_stats={
                    "epoch": epoch_num + 1,
                    "train_loss": epoch_train_loss, 
                    "train_sup_acc": epoch_train_sup_acc,
                    "train_sub_acc": epoch_train_sub_acc,
                    "val_loss": epoch_val_loss,
                    "val_sup_acc": epoch_val_sup_acc,
                    "val_sub_acc": epoch_val_sub_acc
                }
            )

        elif (epoch_num + 1) % 5 == 0:
            chpt_state = model.state_dict()
            
            if args.use_arc_margin or args.use_sub_center_arc_margin:         
                # Keep scale for inference
                chpt_state["arc_margin_scale"] = arc_margin.s
            
            if args.use_sub_center_arc_margin:
                # Keep K_centers for inference
                chpt_state["arc_margin_k_centers"] = arc_margin.K
                
            # Save the intermediate checkpoints just in case
            utils.save_checkpoint(
                save_folder=args.out_folder,
                model_state=chpt_state,
                optimizer_state=feat_optimizer.state_dict(),
                center_loss_optimizer_state=center_loss_optimizer.state_dict() if args.use_center_loss else None,
                training_stats={
                    "epoch": epoch_num + 1,
                    "train_loss": epoch_train_loss, 
                    "train_sup_acc": epoch_train_sup_acc,
                    "train_sub_acc": epoch_train_sub_acc,
                    "val_loss": epoch_val_loss,
                    "val_sup_acc": epoch_val_sup_acc,
                    "val_sub_acc": epoch_val_sub_acc
                },
                epoch=epoch_num + 1 # Save as intermediate checkpoint
            )
        print("\n")


if __name__ == "__main__":
    args = parse_args()
    train(args)
