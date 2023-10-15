import pathlib
import subprocess
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import PIL
import torch
from torch.utils.data import Dataset


def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = PIL.Image.open(path)
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32)/255
    return img.copy()


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg = PIL.Image.open(path)
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg)
    seg = seg.astype(np.float32)
    return seg.copy()


def load_folder(path: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    for file in sorted(path.glob("images/*.png")):  # Adjust the path based on the STARE dataset structure
        img = process_img(file, size=size)
        seg_file = pathlib.Path(str(file).replace("images", "labels").replace(".png", "_seg.png"))  # Adjust based on STARE's structure
        seg = process_seg(seg_file, size=size)
        data.append((img, seg))
    return data


def require_download_stare():
    dest_folder = pathlib.Path("/tmp/universeg_stare/")

    if not dest_folder.exists():
        # Replace with the actual URL of the STARE dataset
        tar_url = "'https://bj.bcebos.com/paddleseg/dataset/stare/stare.zip'"
        subprocess.run(
            ["curl", tar_url, "--create-dirs", "-o",
                str(dest_folder/'stare.zip'),],
            stderr=subprocess.DEVNULL,
            check=True,
        )

        subprocess.run(
            ["unzip", str(dest_folder/'stare.zip'), '-d', str(dest_folder)],
            stderr=subprocess.DEVNULL,
            check=True,
        )

    return dest_folder


@dataclass
class STAREDataset(Dataset):
    split: Literal["support", "test"]
    label: int
    support_frac: float = 0.7

    def __post_init__(self):
        path = require_download_stare()
        T = torch.from_numpy
        self._data = [(T(x)[None], T(y)) for x, y in load_folder(path)]
        if self.label is not None:
            self._ilabel = self.label
        self._idxs = self._split_indexes()

    def _split_indexes(self):
        rng = np.random.default_rng(42)
        N = len(self._data)
        p = rng.permutation(N)
        i = int(np.floor(self.support_frac * N))
        return {"support": p[:i], "test": p[i:]}[self.split]

    def __len__(self):
        return len(self._idxs)

    def __getitem__(self, idx):
        img, seg = self._data[self._idxs[idx]]
        if self.label is not None:
            seg = (seg == self._ilabel)[None]
        return img, seg
