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