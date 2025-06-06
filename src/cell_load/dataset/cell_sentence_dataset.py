import h5py
import logging
import torch
import torch.utils.data as data
import functools
import numpy as np
from typing import Dict
from .. import utils

log = logging.getLogger(__file__)

class CellSentenceDataset(data.Dataset):
    def __init__(self, cfg, test=False, datasets=None, shape_dict=None, adata=None, adata_name=None) -> None:
        super().__init__()
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
                indptrs = h5f["/X/indptr"]
                start_ptr = indptrs[ds_idx]
                end_ptr = indptrs[ds_idx + 1]
                sub_data = torch.tensor(h5f["/X/data"][start_ptr:end_ptr], dtype=torch.float)
                sub_indices = torch.tensor(h5f["/X/indices"][start_ptr:end_ptr], dtype=torch.int32)
                counts = torch.sparse_csr_tensor(
                    [0],
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

class FilteredGenesCounts(CellSentenceDataset):
    def __init__(self, cfg, test=False, datasets=None, shape_dict=None, adata=None, adata_name=None) -> None:
        super().__init__(cfg, test, datasets, shape_dict, adata, adata_name)
        self.valid_gene_index = {}
        _, self.datasets, self.shapes_dict, self.dataset_path_map, self.dataset_group_map = utils.get_shapes_dict(
            "/home/aadduri/state/h5ad_all.csv"
        )
        emb_cfg = utils.get_embedding_cfg(self.cfg)
        try:
            self.ds_emb_map = torch.load(emb_cfg.ds_emb_mapping, weights_only=False)
        except (FileNotFoundError, IOError):
            self.ds_emb_map = {}
        if adata_name is not None:
            self.datasets.append(adata_name)
            self.shapes_dict[adata_name] = adata.shape
            esm_data = torch.load(emb_cfg.all_embeddings, weights_only=False)
            valid_genes_list = list(esm_data.keys())
            global_pos = {g: i for i, g in enumerate(valid_genes_list)}
            gene_names = np.array(adata.var_names)
            new_mapping = np.array([global_pos.get(g, -1) for g in gene_names])
            if (new_mapping == -1).all():
                gene_names = adata.var["gene_name"].values
                new_mapping = np.array([global_pos.get(g, -1) for g in gene_names])
            self.ds_emb_map[adata_name] = new_mapping
        if utils.get_embedding_cfg(self.cfg).ds_emb_mapping is not None:
            esm_data = torch.load(utils.get_embedding_cfg(self.cfg)["all_embeddings"], weights_only=False)
            valid_genes_list = list(esm_data.keys())
            for name in self.datasets:
                if not utils.is_valid_uuid(name):
                    if adata is None:
                        a = self.dataset_file(name)
                        try:
                            gene_names = np.array(
                                [g.decode("utf-8") for g in a["/var/gene_name"][:]]
                            )
                        except:
                            gene_categories = a["/var/gene_name/categories"][:]
                            gene_codes = np.array(a["/var/gene_name/codes"][:])
                            gene_names = np.array([g.decode("utf-8") for g in gene_categories[gene_codes]])
                        valid_mask = np.isin(gene_names, valid_genes_list)
                        self.valid_gene_index[name] = valid_mask
                    else:
                        gene_names = np.array(adata.var_names)
                        valid_mask = np.isin(gene_names, valid_genes_list)
                        if not valid_mask.any():
                            gene_names = adata.var["gene_name"].values
                            valid_mask = np.isin(gene_names, valid_genes_list)
                        self.valid_gene_index[name] = valid_mask

class CellSentenceCollator(object):
    def __init__(self, cfg, valid_gene_mask=None, ds_emb_mapping_inference=None, is_train=True):
        self.pad_length = cfg.dataset.pad_length
        self.P = cfg.dataset.P
        self.N = cfg.dataset.N
        self.S = cfg.dataset.S
        self.cfg = cfg
        self.training = is_train
        self.use_dataset_info = getattr(cfg.model, "dataset_correction", False)
        self.batch_tabular_loss = getattr(cfg.model, "batch_tabular_loss", False)
        if valid_gene_mask is not None:
            self.valid_gene_mask = valid_gene_mask
            self.dataset_to_protein_embeddings = ds_emb_mapping_inference
        else:
            gene_mask_file = utils.get_embedding_cfg(self.cfg).valid_genes_masks
            if gene_mask_file is not None:
                self.valid_gene_mask = torch.load(gene_mask_file, weights_only=False)
            else:
                self.valid_gene_mask = None
            self.dataset_to_protein_embeddings = torch.load(
                utils.get_embedding_cfg(self.cfg).ds_emb_mapping.format(utils.get_embedding_cfg(self.cfg).size),
                weights_only=False,
            )
        self.global_size = utils.get_embedding_cfg(self.cfg).num
        self.global_to_local = {}
        for dataset_name, ds_emb_idxs in self.dataset_to_protein_embeddings.items():
            ds_emb_idxs = torch.tensor(ds_emb_idxs, dtype=torch.long)
            reverse_mapping = torch.full((self.global_size,), -1, dtype=torch.int64)
            local_indices = torch.arange(ds_emb_idxs.size(0), dtype=torch.int64)
            mask = (ds_emb_idxs >= 0) & (ds_emb_idxs < self.global_size)
            reverse_mapping[ds_emb_idxs[mask]] = local_indices[mask]
            self.global_to_local[dataset_name] = reverse_mapping
        print(len(self.global_to_local))
    def __call__(self, batch):
        num_aug = getattr(self.cfg.model, "num_downsample", 1)
        if num_aug > 1 and self.training:
            batch = [item for item in batch for _ in range(num_aug)]
        batch_size = len(batch)
        batch_sentences = torch.zeros((batch_size, self.pad_length), dtype=torch.int32)
        batch_sentences_counts = torch.zeros((batch_size, self.pad_length))
        masks = torch.zeros((batch_size, self.pad_length), dtype=torch.bool)
        idxs = torch.zeros(batch_size, dtype=torch.int32)
        if self.cfg.loss.name == "tabular":
            Xs = torch.zeros((batch_size, self.pad_length, self.P))
            Ys = torch.zeros((batch_size, self.pad_length, self.N))
            batch_weights = torch.ones((batch_size, self.pad_length))
        else:
            Xs = Ys = batch_weights = None
        dataset_nums = torch.zeros(batch_size, dtype=torch.int32)
        total_counts_all = torch.zeros(batch_size)
        for i, (counts, idx, dataset, dataset_num) in enumerate(batch):
            batch_sentences[i, :counts.shape[1]] = counts.squeeze()
            idxs[i] = idx
            dataset_nums[i] = dataset_num
        return (
            batch_sentences,
            Xs,
            Ys,
            idxs,
            batch_weights,
            masks,
            total_counts_all if getattr(self.cfg.model, "rda", False) else None,
            batch_sentences_counts if getattr(self.cfg.model, "counts", False) else None,
            dataset_nums if self.use_dataset_info else None,
        )

    def __init__(self, cfg, test=False, datasets=None, shape_dict=None, adata=None, adata_name=None) -> None:
        super().__init__()
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
                indptrs = h5f["/X/indptr"]
                start_ptr = indptrs[ds_idx]
                end_ptr = indptrs[ds_idx + 1]
                sub_data = torch.tensor(h5f["/X/data"][start_ptr:end_ptr], dtype=torch.float)
                sub_indices = torch.tensor(h5f["/X/indices"][start_ptr:end_ptr], dtype=torch.int32)
                counts = torch.sparse_csr_tensor(
                    [0],
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

class CellSentenceCollator(object):
    def __init__(self, cfg, valid_gene_mask=None, ds_emb_mapping_inference=None, is_train=True):
        self.pad_length = cfg.dataset.pad_length
        self.P = cfg.dataset.P
        self.N = cfg.dataset.N
        self.S = cfg.dataset.S
        self.cfg = cfg
        self.training = is_train
        self.use_dataset_info = getattr(cfg.model, "dataset_correction", False)
        self.batch_tabular_loss = getattr(cfg.model, "batch_tabular_loss", False)
        if valid_gene_mask is not None:
            self.valid_gene_mask = valid_gene_mask
            self.dataset_to_protein_embeddings = ds_emb_mapping_inference
        else:
            gene_mask_file = utils.get_embedding_cfg(self.cfg).valid_genes_masks
            if gene_mask_file is not None:
                self.valid_gene_mask = torch.load(gene_mask_file, weights_only=False)
            else:
                self.valid_gene_mask = None
            self.dataset_to_protein_embeddings = torch.load(
                utils.get_embedding_cfg(self.cfg).ds_emb_mapping.format(utils.get_embedding_cfg(self.cfg).size),
                weights_only=False,
            )
        self.global_size = utils.get_embedding_cfg(self.cfg).num
        self.global_to_local = {}
        for dataset_name, ds_emb_idxs in self.dataset_to_protein_embeddings.items():
            ds_emb_idxs = torch.tensor(ds_emb_idxs, dtype=torch.long)
            reverse_mapping = torch.full((self.global_size,), -1, dtype=torch.int64)
            local_indices = torch.arange(ds_emb_idxs.size(0), dtype=torch.int64)
            mask = (ds_emb_idxs >= 0) & (ds_emb_idxs < self.global_size)
            reverse_mapping[ds_emb_idxs[mask]] = local_indices[mask]
            self.global_to_local[dataset_name] = reverse_mapping
        print(len(self.global_to_local))
    def __call__(self, batch):
        num_aug = getattr(self.cfg.model, "num_downsample", 1)
        if num_aug > 1 and self.training:
            batch = [item for item in batch for _ in range(num_aug)]
        batch_size = len(batch)
        batch_sentences = torch.zeros((batch_size, self.pad_length), dtype=torch.int32)
        batch_sentences_counts = torch.zeros((batch_size, self.pad_length))
        masks = torch.zeros((batch_size, self.pad_length), dtype=torch.bool)
        idxs = torch.zeros(batch_size, dtype=torch.int32)
        if self.cfg.loss.name == "tabular":
            Xs = torch.zeros((batch_size, self.pad_length, self.P))
            Ys = torch.zeros((batch_size, self.pad_length, self.N))
            batch_weights = torch.ones((batch_size, self.pad_length))
        else:
            Xs = Ys = batch_weights = None
        dataset_nums = torch.zeros(batch_size, dtype=torch.int32)
        total_counts_all = torch.zeros(batch_size)
        for i, (counts, idx, dataset, dataset_num) in enumerate(batch):
            batch_sentences[i, :counts.shape[1]] = counts.squeeze()
            idxs[i] = idx
            dataset_nums[i] = dataset_num
        return (
            batch_sentences,
            Xs,
            Ys,
            idxs,
            batch_weights,
            masks,
            total_counts_all if getattr(self.cfg.model, "rda", False) else None,
            batch_sentences_counts if getattr(self.cfg.model, "counts", False) else None,
            dataset_nums if self.use_dataset_info else None,
        )

class FilteredGenesCounts(CellSentenceDataset):
    def __init__(self, cfg, test=False, datasets=None, shape_dict=None, adata=None, adata_name=None) -> None:
        super().__init__(cfg, test, datasets, shape_dict, adata, adata_name)
        self.valid_gene_index = {}
        _, self.datasets, self.shapes_dict, self.dataset_path_map, self.dataset_group_map = utils.get_shapes_dict(
            "/home/aadduri/state/h5ad_all.csv"
        )
        emb_cfg = utils.get_embedding_cfg(self.cfg)
        try:
            self.ds_emb_map = torch.load(emb_cfg.ds_emb_mapping, weights_only=False)
        except (FileNotFoundError, IOError):
            self.ds_emb_map = {}
        if adata_name is not None:
            self.datasets.append(adata_name)
            self.shapes_dict[adata_name] = adata.shape
            esm_data = torch.load(emb_cfg.all_embeddings, weights_only=False)
            valid_genes_list = list(esm_data.keys())
            global_pos = {g: i for i, g in enumerate(valid_genes_list)}
            gene_names = np.array(adata.var_names)
            new_mapping = np.array([global_pos.get(g, -1) for g in gene_names])
            if (new_mapping == -1).all():
                gene_names = adata.var["gene_name"].values
                new_mapping = np.array([global_pos.get(g, -1) for g in gene_names])
            self.ds_emb_map[adata_name] = new_mapping
        if utils.get_embedding_cfg(self.cfg).ds_emb_mapping is not None:
            esm_data = torch.load(utils.get_embedding_cfg(self.cfg)["all_embeddings"], weights_only=False)
            valid_genes_list = list(esm_data.keys())
            for name in self.datasets:
                if not utils.is_valid_uuid(name):
                    if adata is None:
                        a = self.dataset_file(name)
                        try:
                            gene_names = np.array(
                                [g.decode("utf-8") for g in a["/var/gene_name"][:]]
                            )
                        except:
                            gene_categories = a["/var/gene_name/categories"][:]
                            gene_codes = np.array(a["/var/gene_name/codes"][:])
                            gene_names = np.array([g.decode("utf-8") for g in gene_categories[gene_codes]])
                        valid_mask = np.isin(gene_names, valid_genes_list)
                        self.valid_gene_index[name] = valid_mask
                    else:
                        gene_names = np.array(adata.var_names)
                        valid_mask = np.isin(gene_names, valid_genes_list)
                        if not valid_mask.any():
                            gene_names = adata.var["gene_name"].values
                            valid_mask = np.isin(gene_names, valid_genes_list)
                        self.valid_gene_index[name] = valid_mask
