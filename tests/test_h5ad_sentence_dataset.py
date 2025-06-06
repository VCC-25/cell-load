import numpy as np
import torch
import pytest
from types import SimpleNamespace
from cell_load.dataset.h5ad_sentence_dataset import H5adSentenceDataset

class DummyAdata:
    def __init__(self, X, var_names=None):
        self.X = X
        self.shape = X.shape
        self.var = {'gene_name': np.array(var_names) if var_names is not None else np.array(['a', 'b', 'c'])}
        self.var_names = np.array(var_names) if var_names is not None else np.array(['a', 'b', 'c'])

@pytest.fixture
def dummy_cfg():
    # Minimal config with required attributes
    cfg = SimpleNamespace()
    cfg.model = SimpleNamespace()
    cfg.model.batch_size = 2
    cfg.dataset = SimpleNamespace()
    cfg.dataset.pad_length = 3
    cfg.dataset.P = 1
    cfg.dataset.N = 1
    cfg.dataset.S = 1
    return cfg

def test_h5ad_sentence_dataset_with_adata(dummy_cfg):
    X = np.array([[1, 2, 3], [4, 5, 6]])
    adata = DummyAdata(X, var_names=['a', 'b', 'c'])
    ds = H5adSentenceDataset(cfg=dummy_cfg, adata=adata, adata_name='dummy')
    assert len(ds) == 2
    counts, idx, dataset, dataset_num = ds[0]
    assert isinstance(counts, torch.Tensor)
    assert counts.shape[1] == 3
    assert dataset == 'dummy'
    assert dataset_num == 0
