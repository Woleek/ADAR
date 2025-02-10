# Adapted from https://github.com/piotrkawa/audio-deepfake-source-tracing

import os
import random
from collections import defaultdict
from pathlib import Path

import librosa
import numpy as np
import torch
from torch.utils.data import Dataset
from src.datasets.utils import WaveformEmphasiser

from typing import Optional


class MLAADBaseDataset(Dataset):
    def __init__(
        self,
        meta_data: dict,
        basepath: str,
        class_mapping: dict,
        sr: int = 16_000,
        sample_length_s: float = 4,
        n_segments: Optional[int] = 4, # number of segments per sample
        max_samples=-1,
        verbose: bool = True,
    ):
        super().__init__()
        self.class_mapping = {k: v[0] for k, v in class_mapping.items()}
        self.items = meta_data
        self.sample_length_s = sample_length_s
        self.n_segments = n_segments
        self.basepath = basepath
        self.sr = sr
        self.verbose = verbose
        self.classes, self.items = self._parse_items()

        # [TEMP] limit the number of samples per class for testing
        if max_samples > 0:
            counts = {k: 0 for k in self.classes}
            new_items = []
            for k in range(len(self.items)):
                if counts[self.items[k]["class_id"]] < max_samples:
                    new_items.append(self.items[k])
                    counts[self.items[k]["class_id"]] += 1

            self.items = new_items

        if self.verbose:
            self._print_initialization_info()

    def _print_initialization_info(self):
        print("\n > DataLoader initialization")
        print(f" | > Number of instances : {len(self.items)}")
        print(f" | > Max sequence length: {self.sample_length_s} seconds")
        if self.n_segments:
            print(f" | > Number of segments per sequence: {self.n_segments}")
        print(f" | > Num Classes: {len(self.classes)}")
        print(f" | > Classes: {self.classes}")

    def load_wav(self, file_path: str) -> np.ndarray:
        audio, sr = librosa.load(file_path, sr=None)
        if sr != self.sr:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.sr)
        return audio

    def _parse_items(self):
        class_to_utters = defaultdict(list)
        for item in self.items:
            path = Path(self.basepath) / item["path"]
            assert os.path.exists(path), f"File does not exist: {path}"
            class_id = self.class_mapping[item["model_name"]]
            class_to_utters[class_id].append(path)

        classes = sorted(class_to_utters.keys())
        new_items = [
            {
                "wav_file_path": Path(self.basepath) / item["path"],
                "class_id": self.class_mapping[item["model_name"]],
            }
            for item in self.items
        ]
        return classes, new_items

    def __len__(self):
        return len(self.items)

    def get_num_classes(self):
        return len(self.classes)

    def get_class_list(self):
        return self.classes

    def __getitem__(self, idx: int) -> dict:
        return self.items[idx]

    def collate_fn(self, batch: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        labels, feats, files = [], [], []
        target_length = int(self.sample_length_s * self.sr)

        for item in batch:
            utter_path = item["wav_file_path"]
            class_id = item["class_id"]
            wav = self.load_wav(utter_path)
            wav = self._process_wav(wav, target_length)
            if self.n_segments is not None:
                feats.append(torch.from_numpy(wav).unsqueeze(0).float())
            else:
                feats.append(torch.from_numpy(wav).float()) # segments are stacked as channels
            labels.append(class_id)
            files.append(item["wav_file_path"])
        return torch.stack(feats), torch.LongTensor(labels), files

    def _process_wav(self, wav: np.ndarray, target_length: int) -> np.ndarray:
        # Non-segmented samples
        if self.n_segments is None:
            if wav.shape[0] >= target_length:
                offset = random.randint(0, wav.shape[0] - target_length)
                wav = wav[offset : offset + target_length]
            else:
                wav = np.pad(wav, (0, max(0, target_length - wav.shape[0])), mode="wrap")
            return wav
        
        # Segmented samples
        else: 
            segment_length = target_length // self.n_segments
            
            if wav.shape[0] >= target_length:
                segment_offsets = sorted(random.sample(range(wav.shape[0] - segment_length + 1), self.n_segments))
                segments = [wav[offset : offset + segment_length] for offset in segment_offsets]
            else:
                wav = np.pad(wav, (0, max(0, target_length - wav.shape[0])), mode="wrap")
                segments = [wav[i * segment_length : (i + 1) * segment_length] for i in range(self.n_segments)]
            return np.stack(segments)
        
class MLAADFDDataset(Dataset):
    def __init__(
        self, 
        path_to_features, 
        part="train", 
        mode="train", 
        max_samples=-1,
        superclass_mapping=None,
    ):
        super().__init__()
        self.path_to_features = path_to_features
        self.part = part
        self.ptf = os.path.join(path_to_features, self.part)
        self.all_files = librosa.util.find_files(self.ptf, ext="pt")
        if mode == "known":
            # keep only known classes seen during training for F1 metrics
            self.all_files = [
                x for x in self.all_files if int(os.path.basename(x).split("_")[1]) < 24
            ]

        if max_samples > 0:
            self.all_files = self.all_files[:max_samples]

        # Determine the set of labels
        # e.g. "someprefix_<GLOBAL_LABEL>_anything.pt"
        self.all_labels = [int(os.path.split(x)[1].split("_")[1]) for x in self.all_files]
        self.labels = sorted(
            set(self.all_labels)
        )
        self.superclass_mapping = superclass_mapping
        
        if self.part == "train":   
            self._calculate_class_weights()
        
        self._print_info()

    def _print_info(self):
        print(f"Directory: {self.ptf}")
        print(f"Found {len(self.all_files)} samples...")
        if self.superclass_mapping is not None:
            unique_sup = set(self.superclass_mapping.values())
            print(f"Using {len(self.labels)} global classes with {len(unique_sup)} superclasses\n")
        else:
            print(f"Using {len(self.labels)} classes\n")
        print(
            "Seen classes: ",
            set([int(os.path.basename(x).split("_")[1]) for x in self.all_files]),
        )
        
    def _calculate_class_weights(self):
        if self.superclass_mapping is None:
            # Purely subclass-based weighting
            class_counts = {label: self.all_labels.count(label) for label in self.labels}
            self.sample_weights = [1.0 / class_counts[y] for y in self.all_labels]
        else:
            # Count samples for each global label
            subclass_counts = {label: self.all_labels.count(label) for label in self.labels}
            
            # Count samples for each superclass
            all_super_labels = [self.superclass_mapping[label] for label in self.all_labels]
            super_labels = sorted(set(all_super_labels))
            superclass_counts = {sup: all_super_labels.count(sup) for sup in super_labels}
            
            # Calculate weights with alpha
            alpha = 0.5
            self.sample_weights = [
                alpha * (1.0 / subclass_counts[label]) +
                (1 - alpha) * (1.0 / superclass_counts[self.superclass_mapping[label]])
                for label in self.all_labels
            ]

    def __len__(self):
        return len(self.all_files)

    def __getitem__(self, idx):
        filepath = self.all_files[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split("_")

        feature_tensor = torch.load(filepath, weights_only=True)
        filename = "_".join(all_info[2:-1])
        label = int(all_info[1])
        if self.superclass_mapping is not None:
            suplabel = self.superclass_mapping[label]
            label = (suplabel, label)

        return feature_tensor, filename, label
        
class MLAADFD_AR_Dataset(Dataset):
    def __init__(
        self, 
        path_to_dataset, 
        empasizer_pre_augmented,
        empasizer_sampling_rate,
        empasizer_musan_path,
        empasizer_rir_path,
        part="train", 
        mode="train", 
        segmented=False, 
        max_samples=-1,
        superclass_mapping=None,
        ):
        super().__init__()
        self.path_to_dataset = path_to_dataset
        self.part = part
        self.segmented = segmented
        self.ptf = os.path.join(path_to_dataset, self.part)
        self.all_files = librosa.util.find_files(self.ptf, ext="wav")
        if mode == "known":
            # keep only known classes seen during training for F1 metrics
            self.all_files = [
                x for x in self.all_files if int(os.path.basename(x).split("_")[1]) < 24
            ]

        if max_samples > 0:
            self.all_files = self.all_files[:max_samples]

        # Determine the set of labels
        # e.g. "someprefix_<GLOBAL_LABEL>_anything.wav"
        self.all_labels = [int(os.path.split(x)[1].split("_")[1]) for x in self.all_files]
        self.labels = sorted(
            set(self.labels)
        )
            
        self.superclass_mapping = superclass_mapping
        
        
        # Add emphasized data to the dataset
        if not empasizer_pre_augmented:
            self.list_of_emphases = ["original", "reverb", "speech", "music", "noise"]
        else:
            self.list_of_emphases = ["original"]
        
        self.emphasiser = WaveformEmphasiser(
            empasizer_sampling_rate, 
            empasizer_musan_path,
            empasizer_rir_path,
            segmented=segmented
        )
        
        self.all_files_emphasized = []
        self.labels_emphasized = []
        for filepath in self.all_files:
            # Extend labels times number of emphases
            basename = os.path.basename(filepath)
            all_info = basename.split("_")
            label = int(all_info[1])
            self.labels_emphasized.extend([label]*len(self.list_of_emphases))
            
            # Extend files with emphasized versions
            for emphasis in self.list_of_emphases:
                self.all_files_emphasized.append((filepath, emphasis))
             
        if self.part == "train":   
            self._calculate_class_weights()
                
        self._print_info()

    def _print_info(self):
        print(f"Directory: {self.ptf}")
        print(f"Found {len(self.all_files)} samples...")
        print(f"Applied {len(self.list_of_emphases)} emphases...")
        print(f"Resulting in {len(self.all_files_emphasized)} samples...")
        if self.superclass_mapping is not None:
            unique_sup = set(self.superclass_mapping.values())
            print(f"Using {len(self.labels)} global classes with {len(unique_sup)} superclasses\n")
        else:
            print(f"Using {len(self.labels)} classes\n")
        print(
            "Seen classes: ",
            set([int(os.path.basename(x).split("_")[1]) for x in self.all_files]),
        )
        
    def _calculate_class_weights(self):
        if self.superclass_mapping is None:
            # Purely subclass-based weighting
            class_counts = {label: self.all_labels.count(label) for label in self.labels}
            self.sample_weights = [1.0 / class_counts[y] for y in self.all_labels]
        else:
            # Count samples for each global label
            subclass_counts = {label: self.all_labels.count(label) for label in self.labels}
            
            # Count samples for each superclass
            all_super_labels = [self.superclass_mapping[label] for label in self.all_labels]
            super_labels = sorted(set(all_super_labels))
            superclass_counts = {sup: all_super_labels.count(sup) for sup in super_labels}
            
            # Calculate weights with alpha
            alpha = 0.5
            self.sample_weights = [
                alpha * (1.0 / subclass_counts[label]) +
                (1 - alpha) * (1.0 / superclass_counts[self.superclass_mapping[label]])
                for label in self.all_labels
            ]
        
    def load_wav(self, file_path: str) -> np.ndarray:
        if self.segmented:
            audio, sr = librosa.load(file_path, sr=None, mono=False) # segments are stored as channels
        else:
            audio, sr = librosa.load(file_path, sr=None, mono=True)
            audio = np.expand_dims(audio, axis=0)
        if sr != self.emphasiser.sampling_rate:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=self.emphasiser.sampling_rate)
        return audio

    def __len__(self):
        return len(self.all_files_emphasized)

    def __getitem__(self, idx):
        filepath, emphasis = self.all_files_emphasized[idx]
        basename = os.path.basename(filepath)
        all_info = basename.split("_")
        waveform = self.load_wav(filepath)
        feat = torch.from_numpy(waveform).float()
        feat = self.emphasiser(feat, emphasis)
        filename = "_".join(all_info[2:-1])
        label = int(all_info[1])
        if self.superclass_mapping is not None:
            suplabel = self.superclass_mapping[label]
            label = (suplabel, label)

        return feat, filename, label