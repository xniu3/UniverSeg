"""
OASIS dataset processed at https://github.com/adalca/medical-datasets/blob/master/neurite-oasis.md
"""

import pathlib
import subprocess
from dataclasses import dataclass
from typing import Literal, Tuple

import numpy as np
import nibabel as nib
import PIL
import torch
from torch.utils.data import Dataset


def process_img(path: pathlib.Path, size: Tuple[int, int]):
    img = (nib.load(path).get_fdata() * 255).astype(np.uint8).squeeze()
    img = PIL.Image.fromarray(img)
    img = img.resize(size, resample=PIL.Image.BILINEAR)
    img = img.convert("L")
    img = np.array(img)
    img = img.astype(np.float32)/255
    img = np.rot90(img, -1)
    return img.copy()


def process_seg(path: pathlib.Path, size: Tuple[int, int]):
    seg = nib.load(path).get_fdata().astype(np.int8).squeeze()
    seg = PIL.Image.fromarray(seg)
    seg = seg.resize(size, resample=PIL.Image.NEAREST)
    seg = np.array(seg)
    seg = seg.astype(np.float32)
    seg = np.rot90(seg, -1)
    return seg.copy()


def load_folder(path: pathlib.Path, size: Tuple[int, int] = (128, 128)):
    data = []
    for file in sorted(path.glob("*/slice_norm.nii.gz")):
        img = process_img(file, size=size)
        seg_file = pathlib.Path(str(file).replace("slice_norm", "slice_seg24"))
        seg = process_seg(seg_file, size=size)
        data.append((img, seg))
    return data


def require_download_KiTs23():
    dest_folder = pathlib.Path("/tmp/universeg_kits23/")

    if not dest_folder.exists():
        tar_url = "https://kits23-data.s3.us-east-2.amazonaws.com/repo-tarballs/kits23-v0.1.2.tar"
        subprocess.run(
            ["curl", tar_url, "--create-dirs", "-o",
                str(dest_folder/'kits23-v0.1.2.tar'),],
            stderr=subprocess.DEVNULL,
            check=True,
        )

        subprocess.run(
            ["tar", 'xf', str(
                dest_folder/'kits23-v0.1.2.tar'), '-C', str(dest_folder)],
            stderr=subprocess.DEVNULL,
            check=True,
        )

    return dest_folder


@dataclass
class KiTs23Dataset(Dataset):
    split: Literal["support", "test"]
    label: int
    support_frac: float = 0.7

    def __post_init__(self):
        path = require_download_KiTs23()
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
