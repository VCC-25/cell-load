import h5py
import logging
import torch
import torch.utils.data as data
import functools
import numpy as np
from typing import Dict
from .. import utils

log = logging.getLogger(__file__)

EXPONENTIATED_UMIS_LIMIT = 5_000_000
RAW_COUNT_HEURISTIC_THRESHOLD = 35

class H5adSentenceDataset(data.Dataset):
    def __init__(self, cfg, test=False, datasets=None, shape_dict=None, adata=None, adata_name=None) -> None:
        super(H5adSentenceDataset, self).__init__()

        self.adata = None
        self.adata_name = adata_name
        self.test = test
        if adata is not None:
            self.adata = adata
            self.datasets = [adata_name]
            self.shapes_dict = {self.datasets[0]: adata.shape}
        elif datasets is None:
            ds_path = utils.get_dataset_cfg(cfg).train
            if test:
                ds_path = utils.get_dataset_cfg(cfg).val
            _, self.datasets, self.shapes_dict, self.dataset_path_map, self.dataset_group_map = utils.get_shapes_dict(
                ds_path, utils.get_dataset_cfg(cfg).get("filter_by_species")
            )
        else:
            assert shape_dict is not None
            assert len(datasets) == len(shape_dict)
            self.datasets = datasets
            self.shapes_dict = shape_dict
            self.dataset_path_map = {dataset: dataset for dataset in datasets}

        self.datasets = sorted(self.datasets)
        self.cfg = cfg

        self.num_cells = {}
        self.num_genes = {}

        self.total_num_cells = 0
        for name in self.datasets:
            num_cells, num_genes = self.shapes_dict[name]
            self.num_cells[name] = num_cells
            self.num_genes[name] = num_genes
            self.total_num_cells += num_cells

        self.datasets_to_num = {k: v for k, v in zip(self.datasets, range(len(self.datasets)))}

    @functools.lru_cache
    def dataset_file(self, dataset):
        datafile = self.dataset_path_map[dataset]
        return h5py.File(datafile, "r")

    def _compute_index(self, idx):
        for dataset in self.datasets:
            if idx < self.num_cells[dataset]:
                return dataset, idx
            else:
                idx -= self.num_cells[dataset]
        raise IndexError

    def __getitem__(self, idx):
        if self.adata is not None:
            # block is only used during validation
            # if .X is a numpy.ndarray
            if isinstance(self.adata.X, np.ndarray):
                counts = torch.tensor(self.adata.X[idx]).reshape(1, -1)
            else:
                counts = torch.tensor(self.adata.X[idx].todense())

            dataset = self.adata_name
            dataset_num = 0
            return counts, idx, dataset, dataset_num

        dataset, ds_idx = self._compute_index(idx)
        h5f = self.dataset_file(dataset)
        attrs = dict(h5f["X"].attrs)
        try:
            if attrs["encoding-type"] == "csr_matrix":
                indptr = h5f["X"].indptr
                indices = h5f["X"].indices
                data_ = h5f["X"].data
                start = indptr[ds_idx]
                end = indptr[ds_idx + 1]
                sub_indices = torch.tensor(indices[start:end], dtype=torch.int64)
                sub_data = torch.tensor(data_[start:end], dtype=torch.float32)
                counts = torch.sparse_csr_tensor(
                    [0, sub_indices.shape[0]],
                    sub_indices,
                    sub_data,
                    (1, self.num_genes[dataset]),
                )
                counts = counts.to_dense()
            else:
                log.info(ds_idx)
                counts = torch.tensor(h5f["X"][ds_idx]).unsqueeze(0)

        except Exception as iex:
            log.exception(f"Error in dataset {dataset} at index {ds_idx}")
            raise iex

        dataset_num = self.datasets_to_num[dataset]
        return counts, idx, dataset, dataset_num

    def __len__(self) -> int:
        return self.total_num_cells

    def get_dim(self) -> Dict[str, int]:
        return self.num_genes
