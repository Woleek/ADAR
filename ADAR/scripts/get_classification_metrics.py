# Adapted from https://github.com/piotrkawa/audio-deepfake-source-tracing

import argparse
import json
import os
import sys
from pathlib import Path

# Enables running the script from root directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm
from src import utils
from src.datasets.utils import HuggingFaceFeatureExtractor

from src.datasets.dataset import MLAADFD_AR_Dataset
from src.models.w2v2_aasist import W2VAASIST, W2VAASIST_LCL, W2VAASIST_LCPN

import math
def scale(logits, s):
    n_classess = logits.size(1)
    if s == "auto":
        return logits * math.sqrt(2) * math.log(n_classess - 1)
    else:
        return logits * s
    
@ torch.no_grad()
def run_flat_inference(model, models_args, batch, K, s, feature_extractor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat, _, labels = batch
    
    if models_args['is_segmented']:
        n_segments = feat.shape[1]
        feat = feat.view(-1, models_args['sampling_rate']) # [batch_size * num_segments, sampling_rate]
        
    with torch.no_grad():
        feat = feature_extractor(feat, models_args['sampling_rate']).float()
    
    if models_args['is_segmented']:
        feat = feat.view(-1, n_segments, *feat.shape[1:]) # [batch_size, num_segments, *embed_dims]
        feat = feat.mean(dim=1) # [batch_size, *embed_dims] average over segments
            
    feat = feat.transpose(1, 2).to(device)

    _, logits = model(feat)
    
    if models_args['use_sub_center_arc_margin']:
        # Aggregate sub-center outputs
        if K > 1:
            logits = torch.reshape(logits, (-1, models_args['num_classes'], K))
            logits, _ = torch.max(logits, axis=2)
            logits = scale(logits, s)
            
    elif models_args['use_arc_margin']:
        logits = scale(logits, s)
    
    scores = F.softmax(logits, dim=1)
    preds = torch.argmax(scores, dim=1)
    
    return preds, labels

@ torch.no_grad()
def run_hier_inference(model, models_args, batch, K, s, feature_extractor):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    feat, _, labels = batch
    
    if models_args['is_segmented']:
        n_segments = feat.shape[1]
        feat = feat.view(-1, models_args['sampling_rate']) # [batch_size * num_segments, sampling_rate]
        
    with torch.no_grad():
        feat = feature_extractor(feat, models_args['sampling_rate']).float()
    
    if models_args['is_segmented']:
        feat = feat.view(-1, n_segments, *feat.shape[1:]) # [batch_size, num_segments, *embed_dims]
        feat = feat.mean(dim=1) # [batch_size, *embed_dims] average over segments
            
    feat = feat.transpose(1, 2).to(device)

    _, logits = model(feat)
    sup_logits, sub_logits = logits
    
    if models_args['use_sub_center_arc_margin']:
        bs = sup_logits.size(0)
        
        # Aggregate sub-center outputs
        if K > 1:
            sup_logits = torch.reshape(sup_logits, (bs, -1, K))
            sub_logits = torch.reshape(sub_logits, (bs, -1, K))
            
            sup_logits, _ = torch.max(sup_logits, axis=2)
            sub_logits, _ = torch.max(sub_logits, axis=2)
            
            sup_logits = scale(sup_logits, s)
            sub_logits = scale(sub_logits, s)
            
    elif models_args['use_arc_margin']:
        sup_logits = scale(sup_logits, s)
        sub_logits = scale(sub_logits, s)
    
    sup_scores = F.softmax(sup_logits, dim=1)
    sub_scores = F.softmax(sub_logits, dim=1)
    
    sup_preds = torch.argmax(sup_scores, dim=1)
    sub_preds = torch.argmax(sub_scores, dim=1)
    
    if models_args['hierarchy_type'] == 'LCPN':
        global_preds = []
        for sup, sub in zip(sup_preds, sub_preds):
            global_pred = model.get_global_label(sup.item(), sub.item())
            global_preds.append(global_pred)
        sub_preds = torch.tensor(global_preds)
    
    preds = (sup_preds, sub_preds) 
    
    return preds, labels


def parse_args():
    parser = argparse.ArgumentParser(description="Get metrics")
    parser.add_argument(
        "--model_path",
        type=str,
        default="exp/trained_models/base/anti-spoofing_feat_model.pt",
        help="Path to trained model",
    )
    parser.add_argument(
        "-d",
        "--path_to_dataset",
        type=str,
        default="data/prepared_ds/",
        help="Path to the dataset",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for inference"
    )
    parser.add_argument(
        "--superclass_lut",
        type=str,
        default="superclass_mapping_known.csv",
        help="File with superclass mapping for hierarchical classification",
    )
    args = parser.parse_args()
    if not os.path.exists((args.model_path)):
        raise ValueError(f"Model path {args.model_path} does not exist")
    return args


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Running on {device}..")
    
    feature_extractor = HuggingFaceFeatureExtractor(
        model_class_name="Wav2Vec2Model",
        layer=5,
        name="facebook/wav2vec2-base"
    )
    
    # Freeze the feature extractor
    for param in feature_extractor.model.parameters():
        param.requires_grad = False
        
    feature_extractor.model.eval()

    print(f"Loading model from {args.model_path}")
    with open(Path(args.model_path).parent / "args.json", "r") as f:
        model_args = json.load(f) 
    hierarchy = model_args.get("hierarchy_type", None)
    
    state_dict = torch.load(args.model_path, weights_only=True)
    s = state_dict.pop('arc_margin_scale', 1.0)
    K = state_dict.pop('arc_margin_k_centers', 1)
    
    if hierarchy is None:
        id_map = None
        
        # Load flat model
        model = W2VAASIST(
            feature_dim=model_args['feat_dim'], 
            num_labels=model_args['num_classes'] * K if (K > 1 and model_args['use_sub_center_arc_margin']) else model_args['num_classes'],
            normalize_before_output=True if (model_args['use_sub_center_arc_margin'] or model_args['use_arc_margin']) else False,
        )
        
    elif hierarchy == "LCL":
        id_map, _ = utils.load_superclass_mapping(args.path_to_dataset, args.superclass_lut)
        
        num_sup_classes = len(set(id_map.values())) # Number of superclasses
        num_sub_classes = model_args['num_classes']
        
        model = W2VAASIST_LCL(
            feature_dim=model_args['feat_dim'], 
            num_suplabels=num_sup_classes, 
            num_labels=num_sub_classes,
            normalize_before_output=True if (model_args['use_sub_center_arc_margin'] or model_args['use_arc_margin']) else False,
        )
        
    elif hierarchy == "LCPN":
        id_map, _ = utils.load_superclass_mapping(args.path_to_dataset, args.superclass_lut)
        
        model = W2VAASIST_LCPN(
            feature_dim=model_args['feat_dim'], 
            label_mapping=id_map,
            normalize_before_output=True if (model_args['use_sub_center_arc_margin'] or model_args['use_arc_margin']) else False,
            K=K,
        )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Read the data
    dev_dataset = MLAADFD_AR_Dataset(
        args.path_to_dataset, part="dev", mode="known", max_samples=-1, superclass_mapping=id_map,
        empasizer_pre_augmented=model_args['sampling_rate'],
        empasizer_sampling_rate=model_args['sampling_rate'],
        empasizer_musan_path=model_args['sampling_rate'],
        empasizer_rir_path=model_args['sampling_rate'],
        segmented=model_args['is_segmented'],
    )
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, num_workers=0)

    eval_dataset = MLAADFD_AR_Dataset(
        args.path_to_dataset, part="eval", mode="known", max_samples=-1, superclass_mapping=id_map,
        empasizer_pre_augmented=model_args['sampling_rate'],
        empasizer_sampling_rate=model_args['sampling_rate'],
        empasizer_musan_path=model_args['sampling_rate'],
        empasizer_rir_path=model_args['sampling_rate'],
        segmented=model_args['is_segmented'],
    )
    eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=0)

    if len(eval_dataset) == 0:
        print("No data found for evaluation! Exiting...")
        exit(1)

    print("Running on dev data...")
    if hierarchy is None:
        with torch.no_grad():
            all_predicted = np.zeros(len(dev_dataset), dtype=int)
            all_labels = np.zeros(len(dev_dataset), dtype=int)

            dev_bar = tqdm(dev_loader, desc=f"Evaluation")
            for idx, batch in enumerate(dev_bar):
                sample_number = idx * args.batch_size
                
                predicted, labels = run_flat_inference(model, model_args, batch, K, s, feature_extractor)
                predicted = predicted.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
        
                all_predicted[sample_number : sample_number + labels.shape[0]] = predicted
                all_labels[sample_number : sample_number + labels.shape[0]] = labels
    else:
        with torch.no_grad():
            all_predicted = np.zeros(len(dev_dataset), dtype=int)
            all_labels = np.zeros(len(dev_dataset), dtype=int)
            
            all_predicted_sup = np.zeros(len(dev_dataset), dtype=int)
            all_labels_sup = np.zeros(len(dev_dataset), dtype=int)
            
            dev_bar = tqdm(dev_loader, desc=f"Evaluation")
            for idx, batch in enumerate(dev_bar):
                sample_number = idx * args.batch_size
                
                predicted, labels = run_hier_inference(model, model_args, batch, K, s, feature_extractor)
                sup_predicted, sub_predicted = predicted
                sup_predicted = sup_predicted.detach().cpu().numpy()
                sub_predicted = sub_predicted.detach().cpu().numpy()
                sup_labels, sub_labels = labels
                sup_labels = sup_labels.detach().cpu().numpy()
                sub_labels = sub_labels.detach().cpu().numpy()
                
                all_predicted[sample_number : sample_number + sub_labels.shape[0]] = sub_predicted
                all_labels[sample_number : sample_number + sub_labels.shape[0]] = sub_labels
                
                all_predicted_sup[sample_number : sample_number + sup_labels.shape[0]] = sup_predicted
                all_labels_sup[sample_number : sample_number + sup_labels.shape[0]] = sup_labels
            
    print("Classification report for DEV data: ")
    if hierarchy is None:
        report_path = Path(args.model_path).parent/ "dev_in_domain_results.txt"
        report = classification_report(
            all_labels, all_predicted, labels=np.unique(all_labels), zero_division=1.0
        )
        with open(report_path, "w") as f:
            f.write(report)
        print(report)
        print(f"... also written to {report_path}")
    else:
        report_path = Path(args.model_path).parent/ "dev_in_domain_results.txt"
        report = classification_report(
            all_labels, all_predicted, labels=np.unique(all_labels), zero_division=1.0
        )
        with open(report_path, "w") as f:
            f.write(report)
        print(report)
        print(f"... also written to {report_path}")
        
        report_sup_path = Path(args.model_path).parent/ "dev_in_domain_results_sup.txt"
        report_sup = classification_report(
            all_labels_sup, all_predicted_sup, labels=np.unique(all_labels_sup), zero_division=1.0
        )
        with open(report_sup_path, "w") as f:
            f.write(report_sup)
        print(report_sup)
        print(f"... also written to {report_sup_path}")

    print("Running on evaluation data...")
    if hierarchy is None:
        with torch.no_grad():
            all_predicted = np.zeros(len(eval_dataset), dtype=int)
            all_labels = np.zeros(len(eval_dataset), dtype=int)

            eval_bar = tqdm(eval_loader, desc=f"Evaluation")
            for idx, batch in enumerate(eval_bar):
                sample_number = idx * args.batch_size
                
                predicted, labels = run_flat_inference(model, model_args, batch, K, s, feature_extractor)
                predicted = predicted.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()
        
                all_predicted[sample_number : sample_number + labels.shape[0]] = predicted
                all_labels[sample_number : sample_number + labels.shape[0]] = labels
    else:
        with torch.no_grad():
            all_predicted = np.zeros(len(eval_dataset), dtype=int)
            all_labels = np.zeros(len(eval_dataset), dtype=int)
            
            all_predicted_sup = np.zeros(len(eval_dataset), dtype=int)
            all_labels_sup = np.zeros(len(eval_dataset), dtype=int)
            
            eval_bar = tqdm(eval_loader, desc=f"Evaluation")
            for idx, batch in enumerate(eval_bar):
                sample_number = idx * args.batch_size
                
                predicted, labels = run_hier_inference(model, model_args, batch, K, s, feature_extractor)
                sup_predicted, sub_predicted = predicted
                sup_predicted = sup_predicted.detach().cpu().numpy()
                sub_predicted = sub_predicted.detach().cpu().numpy()
                sup_labels, sub_labels = labels
                sup_labels = sup_labels.detach().cpu().numpy()
                sub_labels = sub_labels.detach().cpu().numpy()
                
                all_predicted[sample_number : sample_number + sub_labels.shape[0]] = sub_predicted
                all_labels[sample_number : sample_number + sub_labels.shape[0]] = sub_labels
                
                all_predicted_sup[sample_number : sample_number + sup_labels.shape[0]] = sup_predicted
                all_labels_sup[sample_number : sample_number + sup_labels.shape[0]] = sup_labels

    print("Classification report for EVAL data:")
    if hierarchy is None:
        report_path = Path(args.model_path).parent/"eval_in_domain_results.txt"
        report = classification_report(
            all_labels, all_predicted, labels=np.unique(all_labels), zero_division=1.0
        )
        with open(report_path, "w") as f:
            f.write(report)
        print(report)
        print(f"... also written to {report_path}")
    else:
        report_path = Path(args.model_path).parent/"eval_in_domain_results.txt"
        report = classification_report(
            all_labels, all_predicted, labels=np.unique(all_labels), zero_division=1.0
        )
        with open(report_path, "w") as f:
            f.write(report)
        print(report)
        print(f"... also written to {report_path}")
        
        report_sup_path = Path(args.model_path).parent/"eval_in_domain_results_sup.txt"
        report_sup = classification_report(
            all_labels_sup, all_predicted_sup, labels=np.unique(all_labels_sup), zero_division=1.0
        )
        with open(report_sup_path, "w") as f:
            f.write(report_sup)
        print(report_sup)
        print(f"... also written to {report_sup_path}")


if __name__ == "__main__":
    args = parse_args()
    main(args)
