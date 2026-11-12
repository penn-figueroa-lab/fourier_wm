"""Read selected HDF5 images without decoding discarded intermediate frames."""

import numpy as np
import torch
from stable_worldmodel.data.dataset import Dataset
from stable_worldmodel.data.formats.hdf5 import HDF5Dataset


def load_dataset(name, **kwargs):
    """Keep upstream resolution, metadata, caches and non-HDF5 readers intact."""
    import stable_worldmodel as swm

    raw = swm.data.load_dataset(name, **kwargs)
    return SelectedFrameDataset(raw) if isinstance(raw, HDF5Dataset) else raw


class SelectedFrameDataset(Dataset):
    """Proxy with upstream clip indexing and worker-local HDF5 handles.

    Only image reads change. Actions stay dense; other columns keep upstream
    slicing. Explicit image indices are episode-relative and may repeat for
    causal history padding. They do not change the action/proprioception span.
    """

    def __init__(self, raw):
        self.raw = raw

    def __getattr__(self, name):
        # Pickle may query attributes before restoring the wrapped reader.
        raw = self.__dict__.get("raw")
        if raw is None:
            raise AttributeError(name)
        return getattr(raw, name)

    @property
    def column_names(self):
        return self.raw.column_names

    @property
    def transform(self):
        return self.raw.transform

    @transform.setter
    def transform(self, value):
        self.raw.transform = value

    def _load_slice(self, ep_idx, start, end, *, pixel_indices=None):
        raw = self.raw
        raw._open()
        offset = int(raw.offsets[ep_idx])
        begin, finish = offset + start, offset + end
        steps = {}
        for col in raw.column_names:
            source = raw._cache[col] if col in raw._cache else raw.h5_file[col]
            if col == "pixels" and (pixel_indices is not None or raw.frameskip > 1):
                indices = (
                    range(start, end, raw.frameskip) if pixel_indices is None else pixel_indices
                )
                # Individual rows outperform HDF5 strided hyperslabs on the
                # released compressed datasets. Never use source[::frameskip].
                rows = [source[offset + int(i)] for i in indices]
                data = np.stack(rows) if rows else source[begin:begin]
            else:
                data = source[begin:finish]
                if col != "action":
                    data = data[:: raw.frameskip]
            if data.dtype == np.object_ or data.dtype.kind in ("S", "U"):
                value = data[0] if len(data) else b""
                steps[col] = value.decode() if isinstance(value, bytes) else value
            else:
                steps[col] = torch.from_numpy(data)
                if data.ndim == 4 and data.shape[-1] in (1, 3):
                    steps[col] = steps[col].permute(0, 3, 1, 2)
        return self.transform(steps) if self.transform else steps

    def get_col_data(self, col):
        return self.raw.get_col_data(col)

    def get_dim(self, col):
        return self.raw.get_dim(col)

    def get_row_data(self, row_idx):
        return self.raw.get_row_data(row_idx)

    def merge_col(self, source, target, dim=-1):
        return self.raw.merge_col(source, target, dim)
