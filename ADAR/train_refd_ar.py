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
from src.datasets.dataset import MLAADFD_AR_Dataset
from src.models.w2v2_aasist import W2VAASIST_AR
from src.lossess import ArcMarginProduct, CenterLoss


def parse_args():
    parser = argparse.ArgumentParser("Training script parameters")

    # Paths to features and output
    parser.add_argument(
        "-f",
        "--path_to_dataset",
        type=str,
        default="./exp/prepared_ds/",
        help="Path to the previuosly extracted features",
    )
    parser.add_argument(
        "--is_segmented",
        type=bool,
        default=False,
        help="If audio samples in dataset are split into segments",
    )
    parser.add_argument(
        "--out_folder", type=str, default="./exp/trained_models/base", help="Output folder"
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
        choices=["ce"], # "bce"
        help="Loss for basic training",
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
    parser.add_argument("--arc_s", type=float_or_str, default="auto", help="Scale parameter s for ArcMarginProduct")
    parser.add_argument("--arc_m", type=float, default=0.05, help="Margin parameter m for ArcMarginProduct")
    parser.add_argument("--easy_margin", type=bool, default=False, help="Use easy margin in ArcMarginProduct")
    parser.add_argument("--optimize_arc_margin_weights", type=bool, default=True, help="Optimize ArcMarginProduct weights")
    
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
    # Load the train and dev data (only known classes)
    print("Loading training data...")
    training_set = MLAADFD_AR_Dataset(args.path_to_dataset, "train", segmented=args.is_segmented, emphasiser_args=args)
    print("\nLoading dev data...")
    dev_set = MLAADFD_AR_Dataset(args.path_to_dataset, "dev", mode="known", segmented=args.is_segmented, emphasiser_args=args)

    train_loader = DataLoader(
        training_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=torch_sampler.SubsetRandomSampler(range(len(training_set))),
    )
    dev_loader = DataLoader(
        dev_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        sampler=torch_sampler.SubsetRandomSampler(range(len(dev_set))),
    )
    
    start_epoch = 0

    # Setup the model to learn in-domain classess
    model = W2VAASIST_AR(args.feat_dim, args.num_classes, extractor_args=args).to(args.device)
    
    # Set up loss functions
    if args.base_loss == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCELoss()
        
    if args.use_arc_margin:
        print("[INFO] Using ArcFace Loss...")
        arc_margin = ArcMarginProduct(
            in_features=5 * 32, # Last hidden output size of W2VAASIST
            out_features=args.num_classes,
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
    
    if args.resume_checkpoint and args.resume_epoch:
        print(f"Resuming training from checkpoint {args.resume_checkpoint} at epoch {args.resume_epoch}")
        model.load_state_dict(torch.load(args.resume_checkpoint, weights_only=True))
        
        start_epoch = int(args.resume_epoch)
        
    # Main optimizer + arc_margin params (optional)
    feat_optimizer = torch.optim.Adam(
        list(model.parameters()) + list(arc_margin.parameters()) if (args.optimize_arc_margin_weights and args.use_arc_margin) else model.parameters(),
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
        train_accuracy, train_loss = [], []
        for iter_num, batch in enumerate(epoch_bar):
            feat, _, labels = batch # audio_sample, path, class_id
            
            if args.is_segmented:
                n_segments = feat.shape[1]
                feat = feat.view(-1, args.sampling_rate) # [batch_size * num_segments, sampling_rate]
                
            feat = model.extract_features(feat)
            
            if args.is_segmented:
                feat = feat.view(-1, n_segments, *feat.shape[1:]) # [batch_size, num_segments, *embed_dims]
                feat = feat.mean(dim=1) # [batch_size, *embed_dims] average over segments
            feat = feat.transpose(1, 2).to(args.device)
            labels = labels.to(args.device)

            if not (args.use_arc_margin or args.use_center_loss): # Not fit for ArcFace and CenterLoss
                # Manifold Mixup - mixes the interpolates embeddings and labels to create a new input and label pairs. The lambda value is used to balance the contribution of the two pairs.
                mix_feat, y_a, y_b, lam = utils.mixup_data( 
                    feat, labels, args.device, alpha=0.5
                )

                targets_a = torch.cat([labels, y_a])
                targets_b = torch.cat([labels, y_b])
                feat = torch.cat([feat, mix_feat], dim=0)

            # ---- Forward pass ----
            feats, base_logits = model(feat) # feats - last hidden, base_logits - model output from linear
            
            # ---- Compute base loss (CE or BCE) ----
            total_loss = 0
            if args.use_arc_margin: # Use arc margin logits instaed
                logits = arc_margin(feats, labels)
                
                # if args.base_loss == "bce":
                #     loss_base = criterion(logits, labels.unsqueeze(1).float())
                # else:
                loss_base = criterion(logits, labels) # ArcMargin expects hard labels so no Mixup
            
            else: # use model original output
                logits = base_logits
                
                # if args.base_loss == "bce":
                #     loss_base = criterion(logits, labels.unsqueeze(1).float())
                if not args.use_center_loss:
                    loss_base = utils.regmix_criterion(
                        criterion, logits, targets_a, targets_b, lam
                    )
                else:
                    loss_base = criterion(logits, labels)
                    
            total_loss += loss_base
                
            # ---- Center loss ----
            if args.use_center_loss:
                loss_center = center_loss_fn(feats, labels) # Center loss expects hard labels so no Mixup
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
                score = F.softmax(logits, dim=1)  # [:, 0]
                predicted = torch.argmax(score, dim=1)
                acc = (predicted == labels).float().mean()

            train_accuracy.append(acc.item())
            train_loss.append(total_loss.item())
            
            epoch_bar.set_postfix(
                {
                    "train_loss": f"{sum(train_loss)/(iter_num+1):.4f}",
                    "acc": f"{sum(train_accuracy)/(iter_num+1):.2f}",
                }
            )
   
        epoch_train_loss = sum(train_loss) / (iter_num + 1)
        epoch_train_acc = sum(train_accuracy) / (iter_num + 1)

        # Epoch eval
        model.eval()
        with torch.no_grad():
            val_bar = tqdm(dev_loader, desc=f"Validation for epoch {epoch_num+1}")
            val_accuracy, val_loss = [], []
            for iter_num, batch in enumerate(val_bar):
                feat, _, labels = batch
                if args.is_segmented:
                    n_segments = feat.shape[1]
                    feat = feat.view(-1, args.sampling_rate)
                
                feat = model.extract_features(feat)
                
                if args.is_segmented:
                    feat = feat.view(-1, n_segments, *feat.shape[1:])
                    feat = feat.mean(dim=1)
                feat = feat.transpose(1, 2).to(args.device)
                labels = labels.to(args.device)

                feats, base_logits = model(feat)
                
                if args.use_arc_margin:
                    logits = arc_margin(feats, labels)
                else:
                    logits = base_logits
                
                # if args.base_loss == "bce":
                #     loss = criterion(logits, labels.unsqueeze(1).float())
                #     score = logits
                # else:
                loss = criterion(logits, labels)
                score = F.softmax(logits, dim=1)
                
                predicted = torch.argmax(score, dim=1)
                acc = (predicted == labels).float().mean()
                
                val_accuracy.append(acc.item())
                val_loss.append(loss.item())
                
                val_bar.set_postfix(
                    {
                        "val_loss": f"{sum(val_loss)/(iter_num+1):.4f}",
                        "val_acc": f"{sum(val_accuracy)/(iter_num+1):.2f}",
                    }
                )

        epoch_val_loss = sum(val_loss) / (iter_num + 1)
        epoch_val_acc = sum(val_accuracy) / (iter_num + 1)
        
        if epoch_val_loss < prev_loss:
            # Save the checkpoint with better val_loss
            checkpoint_path = os.path.join(
                args.out_folder, "anti-spoofing_feat_model.pth"
            )
            print(f"[INFO] Saving model with better val_loss to {checkpoint_path}")
            torch.save(model.state_dict(), checkpoint_path)
            prev_loss = epoch_val_loss
            
            # Save optimizer state
            torch.save(feat_optimizer.state_dict(), os.path.join(args.out_folder, "optimizer.pth"))
            if args.use_center_loss:
                torch.save(center_loss_optimizer.state_dict(), os.path.join(args.out_folder, "center_loss_optimizer.pth"))
            
            # Save training stats
            with open(os.path.join(args.out_folder, "training_stats.json"), "w") as file:
                file.write(json.dumps({
                    "epoch": epoch_num + 1,
                    "train_loss": epoch_train_loss, 
                    "train_acc": epoch_train_acc,
                    "val_loss": epoch_val_loss,
                    "val_acc": epoch_val_acc
                }))

        elif (epoch_num + 1) % 5 == 0:
            # Save the intermediate checkpoints just in case
            checkpoint_path = os.path.join(
                args.out_folder,
                "checkpoint",
                "anti-spoofing_feat_model_%02d.pth" % (epoch_num + 1),
            )
            print(
                f"[INFO] Saving intermediate model at epoch {epoch_num+1} to {checkpoint_path}"
            )
            torch.save(model.state_dict(), checkpoint_path)
            
            # Save optimizer state
            torch.save(feat_optimizer.state_dict(), os.path.join(args.out_folder, "checkpoint", "optimizer_%02d.pth" % (epoch_num + 1)))
            if args.use_center_loss:
                torch.save(center_loss_optimizer.state_dict(), os.path.join(args.out_folder, "checkpoint", "center_loss_optimizer_%02d.pth" % (epoch_num + 1)))
            
            # Save training stats just in case
            with open(os.path.join(args.out_folder, "checkpoint","training_stats_%02d.json" % (epoch_num + 1)), "w") as file:
                file.write(json.dumps({
                    "epoch": epoch_num + 1,
                    "train_loss": epoch_train_loss, 
                    "train_acc": epoch_train_acc,
                    "val_loss": epoch_val_loss,
                    "val_acc": epoch_val_acc
                }))
        print("\n")


if __name__ == "__main__":
    args = parse_args()
    train(args)
