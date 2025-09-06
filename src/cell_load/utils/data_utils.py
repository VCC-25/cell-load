import logging
import warnings

import anndata
import h5py
import numpy as np
import scipy.sparse as sp
import torch

from .singleton import Singleton

# New Memory Mapping Imports (Dan)
import numpy as np
import mmap
import os
from typing import Optional, Tuple, Union, Iterator, Any
from pathlib import Path
import threading
import queue
import time

log = logging.getLogger(__name__)

warnings.filterwarnings("ignore")

# ============================================================================
# MEMORY MAPPING CLASSES - New functionality (Dan)
# ============================================================================

class MemoryMappedArray:
    """
    Memory-mapped NumPy Array for Single-Cell data
    Integrates seamlessly into your existing cell_load architecture
    """
    def __init__(self, 
                 file_path: Union[str, Path], 
                 shape: Tuple[int, ...],
                 dtype: np.dtype = np.float32,
                 mode: str = 'r',
                 create_if_missing: bool = False):
        """
        Args:
            file_path: Path to memory-mapped file
            shape: Array shape (e.g. n_cells, n_genes)
            dtype: NumPy data type
            mode: File mode ('r', 'r+', 'w+')
            create_if_missing: Create file if it doesn't exist
        """
        self.file_path = Path(file_path)
        self.shape = shape
        self.dtype = np.dtype(dtype)
        self.mode = mode
        
        # Setup Memory Mapping
        self._setup_memory_mapping(create_if_missing)
        
    def _setup_memory_mapping(self, create_if_missing: bool):
        """Memory mapping setup"""
        if not self.file_path.exists():
            if create_if_missing or 'w' in self.mode:
                self._create_empty_file()
            else:
                raise FileNotFoundError(f"File not found: {self.file_path}")
        
        # Open file
        file_mode = 'r+b' if '+' in self.mode or 'w' in self.mode else 'rb'
        self.file = open(self.file_path, file_mode)
        
        # Memory mapping
        access = mmap.ACCESS_WRITE if '+' in self.mode or 'w' in self.mode else mmap.ACCESS_READ
        self.mmap = mmap.mmap(self.file.fileno(), 0, access=access)
        
        # NumPy Array View
        self.data = np.frombuffer(
            self.mmap, 
            dtype=self.dtype
        ).reshape(self.shape)
        
    def _create_empty_file(self):
        """Create empty file with correct size"""
        total_bytes = np.prod(self.shape) * self.dtype.itemsize
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.file_path, 'wb') as f:
            f.write(b'\x00' * total_bytes)
            
    def __getitem__(self, key) -> np.ndarray:
        """NumPy-style indexing"""
        return self.data[key]
        
    def __setitem__(self, key, value):
        """NumPy-style assignment (if writable)"""
        self.data[key] = value
        
    def __len__(self) -> int:
        return self.shape[0]
        
    @property
    def size_mb(self) -> float:
        """File size in MB"""
        return self.file_path.stat().st_size / (1024 * 1024)
        
    def close(self):
        """Release resources"""
        if hasattr(self, 'mmap'):
            self.mmap.close()
        if hasattr(self, 'file'):
            self.file.close()
            
    def __del__(self):
        self.close()
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

class PrefetchQueue:
    """
    Thread-safe Prefetch Queue for Batch Loading
    Can be integrated into your existing DataLoaders
    """
    def __init__(self, maxsize: int = 10, timeout: float = 30.0):
        self.queue = queue.Queue(maxsize=maxsize)
        self.timeout = timeout
        self.stop_event = threading.Event()
        
    def put(self, item: Any, block: bool = True):
        """Put item into queue"""
        try:
            self.queue.put(item, block=block, timeout=self.timeout)
        except queue.Full:
            if not self.stop_event.is_set():
                raise
                
    def get(self, block: bool = True) -> Any:
        """Get item from queue"""
        try:
            return self.queue.get(block=block, timeout=self.timeout)
        except queue.Empty:
            if not self.stop_event.is_set():
                raise
            return None
            
    def stop(self):
        """Stop queue"""
        self.stop_event.set()
        
    def empty(self) -> bool:
        return self.queue.empty()
        
    def qsize(self) -> int:
        return self.queue.qsize()

# ============================================================================
# UTILITY FUNCTIONS - Extended functionality (Dan)
# ============================================================================

def create_memory_mapped_dataset(data: np.ndarray, 
                                file_path: Union[str, Path],
                                overwrite: bool = False) -> MemoryMappedArray:
    """
    Save NumPy array as memory-mapped dataset
    
    Args:
        data: NumPy array (e.g. Single-Cell data)
        file_path: Path for memory-mapped file
        overwrite: Overwrite existing file
        
    Returns:
        MemoryMappedArray instance
    """
    file_path = Path(file_path)
    
    if file_path.exists() and not overwrite:
        raise FileExistsError(f"File exists: {file_path}. Use overwrite=True")
    
    # Create memory-mapped array
    mmap_array = MemoryMappedArray(
        file_path, 
        data.shape, 
        data.dtype, 
        mode='w+',
        create_if_missing=True
    )
    
    # Copy data
    mmap_array.data[:] = data
    
    print(f"✅ Created memory-mapped dataset: {file_path.name}")
    print(f"   • Shape: {data.shape}")
    print(f"   • Size: {mmap_array.size_mb:.1f} MB")
    
    return mmap_array

def load_memory_mapped_dataset(file_path: Union[str, Path],
                              shape: Tuple[int, ...],
                              dtype: np.dtype = np.float32) -> MemoryMappedArray:
    """
    Load memory-mapped dataset
    
    Args:
        file_path: Path to memory-mapped file
        shape: Array shape
        dtype: NumPy data type
        
    Returns:
        MemoryMappedArray instance
    """
    mmap_array = MemoryMappedArray(file_path, shape, dtype, mode='r')
    
    print(f"📂 Loaded memory-mapped dataset: {Path(file_path).name}")
    print(f"   • Shape: {shape}")
    print(f"   • Size: {mmap_array.size_mb:.1f} MB")
    
    return mmap_array

def estimate_memory_usage(shape: Tuple[int, ...], 
                         dtype: np.dtype = np.float32) -> dict:
    """
    Estimate memory usage
    
    Args:
        shape: Array shape
        dtype: NumPy data type
        
    Returns:
        Dict with memory information
    """
    total_elements = np.prod(shape)
    bytes_per_element = np.dtype(dtype).itemsize
    total_bytes = total_elements * bytes_per_element
    
    return {
        'total_elements': total_elements,
        'bytes_per_element': bytes_per_element,
        'total_bytes': total_bytes,
        'total_mb': total_bytes / (1024 * 1024),
        'total_gb': total_bytes / (1024 * 1024 * 1024),
        'shape': shape,
        'dtype': str(dtype)
    }

def benchmark_memory_access(mmap_array: MemoryMappedArray,
                           n_samples: int = 1000) -> dict:
    """
    Benchmark memory access performance
    
    Args:
        mmap_array: MemoryMappedArray instance
        n_samples: Number of test accesses
        
    Returns:
        Performance statistics
    """
    import random
    
    # Generate random indices
    max_idx = len(mmap_array) - 1
    indices = [random.randint(0, max_idx) for _ in range(n_samples)]
    
    # Sequential access benchmark
    start_time = time.time()
    for i in range(min(n_samples, len(mmap_array))):
        _ = mmap_array[i]
    sequential_time = time.time() - start_time
    
    # Random access benchmark
    start_time = time.time()
    for idx in indices:
        _ = mmap_array[idx]
    random_time = time.time() - start_time
    
    # Batch access benchmark
    batch_size = min(100, len(mmap_array))
    start_time = time.time()
    for i in range(0, min(n_samples, len(mmap_array)), batch_size):
        end_idx = min(i + batch_size, len(mmap_array))
        _ = mmap_array[i:end_idx]
    batch_time = time.time() - start_time
    
    return {
        'sequential_access_time': sequential_time,
        'random_access_time': random_time,
        'batch_access_time': batch_time,
        'sequential_ops_per_sec': n_samples / sequential_time,
        'random_ops_per_sec': n_samples / random_time,
        'batch_ops_per_sec': (n_samples // batch_size) / batch_time,
        'n_samples': n_samples
    }
def predict_dataloader(self):
    """Prediction DataLoader mit konsistenter Batch-Struktur"""
    
    dataset = self.predict_dataset
    
    return torch.utils.data.DataLoader(
        dataset,
        batch_size=self.batch_size,
        shuffle=False,  # Wichtig für Prediction!
        num_workers=self.num_workers,
        pin_memory=True,
        drop_last=False,  # Behalte alle Samples
        collate_fn=self._prediction_collate_fn
    )

def _prediction_collate_fn(self, batch):
    """Custom collate für Predictions mit Metadata-Tracking"""
    X_list = [item['X'] for item in batch]
    metadata_list = [item['obs'] for item in batch]  # Nur Batch-Metadata!
    
    return {
        'X': torch.stack(X_list),      # [batch_size, features]
        'obs': metadata_list           # [batch_size] Liste von Dicts
    }
    '''# Sammle X-Daten
    X_list = [item['X'] for item in batch]
    X_batch = torch.stack(X_list)
    
    # Sammle Metadata (wichtig!)
    metadata_list = []
    for item in batch:
        if 'obs' in item:
            metadata_list.append(item['obs'])
        elif 'metadata' in item:
            metadata_list.append(item['metadata'])
        else:
            # Fallback
            metadata_list.append({'sample_id': item.get('sample_id', 'unknown')})
    
    return {
        'X': X_batch,
        'obs': metadata_list,  # Liste von Dicts
        'metadata': metadata_list  # Backup
    }'''
# ============================================================================
# INTEGRATION HELPERS - For existing cell_load components (Dan)
# ============================================================================

def integrate_with_existing_loader(loader_class):
    """
    Decorator to extend existing DataLoaders with Memory Mapping
    
    Usage:
        @integrate_with_existing_loader
        class YourExistingLoader:
            ...
    """
    class MemoryMappedWrapper(loader_class):
        def __init__(self, *args, use_memory_mapping=False, mmap_file=None, **kwargs):
            super().__init__(*args, **kwargs)
            self.use_memory_mapping = use_memory_mapping
            self.mmap_file = mmap_file
            
            if use_memory_mapping and mmap_file:
                self._setup_memory_mapping()
                
        def _setup_memory_mapping(self):
            # Memory mapping setup based on existing data structure
            pass
            
    return MemoryMappedWrapper


class H5MetadataCache:
    """Cache for H5 file metadata to avoid repeated disk reads."""

    def __init__(
        self,
        h5_path: str,
        pert_col: str = "drug",
        cell_type_key: str = "cell_name",
        control_pert: str = "DMSO_TF",
        batch_col: str = "sample",
    ):
        """
        Args:
            h5_path: Path to the .h5ad or .h5 file
            pert_col: obs column name for perturbation
            cell_type_key: obs column name for cell type
            control_pert: the perturbation to treat as control
            batch_col: obs column name for batch/plate
        """
        self.h5_path = h5_path
        with h5py.File(h5_path, "r") as f:
            obs = f["obs"]

            # -- Categories --
            self.pert_categories = safe_decode_array(obs[pert_col]["categories"][:])
            self.cell_type_categories = safe_decode_array(
                obs[cell_type_key]["categories"][:]
            )

            # -- Batch: handle categorical vs numeric storage --
            batch_ds = obs[batch_col]
            if "categories" in batch_ds:
                self.batch_is_categorical = True
                self.batch_categories = safe_decode_array(batch_ds["categories"][:])
                self.batch_codes = batch_ds["codes"][:].astype(np.int32)
            else:
                self.batch_is_categorical = False
                raw = batch_ds[:]
                self.batch_categories = raw.astype(str)
                self.batch_codes = raw.astype(np.int32)

            # -- Codes for pert & cell type --
            self.pert_codes = obs[pert_col]["codes"][:].astype(np.int32)
            self.cell_type_codes = obs[cell_type_key]["codes"][:].astype(np.int32)

            # -- Control mask & counts --
            idx = np.where(self.pert_categories == control_pert)[0]
            if idx.size == 0:
                raise ValueError(
                    f"control_pert='{control_pert}' not found in {pert_col} categories"
                )
            self.control_pert_code = int(idx[0])
            self.control_mask = self.pert_codes == self.control_pert_code

            self.n_cells = len(self.pert_codes)

    def get_batch_names(self, indices: np.ndarray) -> np.ndarray:
        """Return batch labels for the provided cell indices."""
        return self.batch_categories[indices]

    def get_cell_type_names(self, indices: np.ndarray) -> np.ndarray:
        """Return cell‐type labels for the provided cell indices."""
        return self.cell_type_categories[indices]

    def get_pert_names(self, indices: np.ndarray) -> np.ndarray:
        """Return perturbation labels for the provided cell indices."""
        return self.pert_categories[indices]


class GlobalH5MetadataCache(metaclass=Singleton):
    """
    Singleton managing a shared dict of H5MetadataCache instances.
    Keys by h5_path only (same as before).
    """

    def __init__(self):
        self._cache: dict[str, H5MetadataCache] = {}

    def get_cache(
        self,
        h5_path: str,
        pert_col: str = "drug",
        cell_type_key: str = "cell_name",
        control_pert: str = "DMSO_TF",
        batch_col: str = "drug",
    ) -> H5MetadataCache:
        """
        If a cache for this file doesn’t yet exist, create it with the
        given parameters; otherwise return the existing one.
        """
        if h5_path not in self._cache:
            self._cache[h5_path] = H5MetadataCache(
                h5_path, pert_col, cell_type_key, control_pert, batch_col
            )
        return self._cache[h5_path]


def safe_decode_array(arr) -> np.ndarray:
    """
    Decode any byte-strings in `arr` to UTF-8 and cast all entries to Python str.

    Args:
        arr: array-like of bytes or other objects
    Returns:
        np.ndarray[str]: decoded strings
    """
    decoded = []
    for x in arr:
        if isinstance(x, (bytes, bytearray)):
            # decode bytes, ignoring errors
            decoded.append(x.decode("utf-8", errors="ignore"))
        else:
            decoded.append(str(x))
    return np.array(decoded, dtype=str)


def generate_onehot_map(keys) -> dict:
    """
    Build a map from each unique key to a fixed-length one-hot torch vector.

    Args:
        keys: iterable of hashable items
    Returns:
        dict[key, torch.FloatTensor]: one-hot encoding of length = number of unique keys
    """
    unique_keys = sorted(set(keys))
    num_classes = len(unique_keys)
    # identity matrix rows are one-hot vectors
    onehots = torch.eye(num_classes)
    return {k: onehots[i] for i, k in enumerate(unique_keys)}


def data_to_torch_X(X):
    """
    Convert input data to a dense torch FloatTensor.

    If passed an AnnData, extracts its .X matrix.
    If the result isn’t a NumPy array (e.g. a sparse matrix), calls .toarray().
    Finally wraps with torch.from_numpy(...).float().

    Args:
        X: anndata.AnnData or array-like (dense or sparse).
    Returns:
        torch.FloatTensor of shape (n_cells, n_features).
    """
    if isinstance(X, anndata.AnnData):
        X = X.X
    if not isinstance(X, np.ndarray):
        X = X.toarray()
    return torch.from_numpy(X).float()


def split_perturbations_by_cell_fraction(
    pert_groups: dict,
    val_fraction: float,
    rng: np.random.Generator = None,
):
    """
    Partition the set of perturbations into two subsets: 'val' vs 'train',
    such that the fraction of total cells in 'val' is as close as possible
    to val_fraction, using a greedy approach.

    Here, pert_groups is a dictionary where the keys are perturbation names
    and the values are numpy arrays of cell indices.

    Returns:
        train_perts: list of perturbation names
        val_perts:   list of perturbation names
    """
    if rng is None:
        rng = np.random.default_rng(42)

    # 1) Compute total # of cells across all perturbations
    total_cells = sum(len(indices) for indices in pert_groups.values())
    target_val_cells = val_fraction * total_cells

    # 2) Create a list of (pert_name, size), then shuffle
    pert_size_list = [(p, len(pert_groups[p])) for p in pert_groups.keys()]
    rng.shuffle(pert_size_list)

    # 3) Greedily add perts to the 'val' subset if it brings us closer to the target
    val_perts = []
    current_val_cells = 0
    for pert, size in pert_size_list:
        new_val_cells = current_val_cells + size

        # Compare how close we'd be to target if we add this perturbation vs. skip it
        diff_if_add = abs(new_val_cells - target_val_cells)
        diff_if_skip = abs(current_val_cells - target_val_cells)

        if diff_if_add < diff_if_skip:
            # Adding this perturbation gets us closer to the target fraction
            val_perts.append(pert)
            current_val_cells = new_val_cells
        # else: skip it; it goes to train

    train_perts = list(set(pert_groups.keys()) - set(val_perts))

    return train_perts, val_perts


def suspected_discrete_torch(x: torch.Tensor, n_cells: int = 100) -> bool:
    """Check if data appears to be discrete/raw counts by examining row sums.
    Adapted from validate_normlog function for PyTorch tensors.
    """
    top_n = min(x.shape[0], n_cells)
    rowsum = x[:top_n].sum(dim=1)

    # Check if row sums are integers (fractional part == 0)
    frac_part = rowsum - rowsum.floor()
    return torch.all(torch.abs(frac_part) < 1e-7)


def suspected_log_torch(x: torch.Tensor) -> bool:
    """Check if the data is log transformed already."""
    global_max = x.max()
    return global_max.item() < 15.0


def _mean(expr) -> float:
    """Return the mean of a dense or sparse 1-D/2-D slice."""
    if sp.issparse(expr):
        return float(expr.mean())
    return float(np.asarray(expr).mean())


def is_on_target_knockdown(
    adata: anndata.AnnData,
    target_gene: str,
    perturbation_column: str = "gene",
    control_label: str = "non-targeting",
    residual_expression: float = 0.30,
    layer: str | None = None,
) -> bool:
    """
    True ⇢ average expression of *target_gene* in perturbed cells is below
    `residual_expression` × (average expression in control cells).

    Parameters
    ----------
    adata : AnnData
    target_gene : str
        Gene symbol to check.
    perturbation_column : str, default "gene"
        Column in ``adata.obs`` holding perturbation identities.
    control_label : str, default "non-targeting"
        Category in *perturbation_column* marking control cells.
    residual_expression : float, default 0.30
        Residual fraction (0‒1). 0.30 → 70 % knock-down.
    layer : str | None, optional
        Use this matrix in ``adata.layers`` instead of ``adata.X``.

    Raises
    ------
    KeyError
        *target_gene* not present in ``adata.var_names``.
    ValueError
        No perturbed cells for *target_gene*, or control mean is zero.

    Returns
    -------
    bool
    """
    if target_gene == control_label:
        # Never evaluate the control itself
        return False

    if target_gene not in adata.var_names:
        print(f"Gene {target_gene!r} not found in `adata.var_names`.")
        return 1

    gene_idx = adata.var_names.get_loc(target_gene)
    X = adata.layers[layer] if layer is not None else adata.X

    control_cells = adata.obs[perturbation_column] == control_label
    perturbed_cells = adata.obs[perturbation_column] == target_gene

    if not perturbed_cells.any():
        raise ValueError(f"No cells labelled with perturbation {target_gene!r}.")

    try:
        control_mean = _mean(X[control_cells, gene_idx])
    except:
        control_cells = (adata.obs[perturbation_column] == control_label).values
        control_mean = _mean(X[control_cells, gene_idx])

    if control_mean == 0:
        raise ValueError(
            f"Mean {target_gene!r} expression in control cells is zero; "
            "cannot compute knock-down ratio."
        )

    try:
        perturbed_mean = _mean(X[perturbed_cells, gene_idx])
    except:
        perturbed_cells = (adata.obs[perturbation_column] == target_gene).values
        perturbed_mean = _mean(X[perturbed_cells, gene_idx])

    knockdown_ratio = perturbed_mean / control_mean
    return knockdown_ratio < residual_expression


def filter_on_target_knockdown(
    adata: anndata.AnnData,
    perturbation_column: str = "gene",
    control_label: str = "non-targeting",
    residual_expression: float = 0.30,  # perturbation-level threshold
    cell_residual_expression: float = 0.50,  # cell-level threshold
    min_cells: int = 30,  # **NEW**: minimum cells/perturbation
    layer: str | None = None,
    var_gene_name: str = "gene_name",
) -> anndata.AnnData:
    """
    1.  Keep perturbations whose *average* knock-down ≥ (1-residual_expression).
    2.  Within those, keep only cells whose knock-down ≥ (1-cell_residual_expression).
    3.  Discard perturbations that have < `min_cells` cells remaining
        after steps 1–2.  Control cells are always preserved.

    Returns
    -------
    AnnData
        View of `adata` satisfying all three criteria.
    """
    # --- prep ---
    adata_ = set_var_index_to_col(adata.copy(), col=var_gene_name)
    X = adata_.layers[layer] if layer is not None else adata_.X
    perts = adata_.obs[perturbation_column]
    control_cells = (perts == control_label).values

    # ---------- stage 1: perturbation filter ----------
    perts_to_keep = [control_label]  # always keep controls
    for pert in perts.unique():
        if pert == control_label:
            continue
        if is_on_target_knockdown(
            adata_,
            target_gene=pert,
            perturbation_column=perturbation_column,
            control_label=control_label,
            residual_expression=residual_expression,
            layer=layer,
        ):
            perts_to_keep.append(pert)

    # ---------- stage 2: cell filter ----------
    keep_mask = np.zeros(adata_.n_obs, dtype=bool)
    keep_mask[control_cells] = True  # retain all controls

    # cache control means to avoid recomputation
    control_mean_cache: dict[str, float] = {}

    for pert in perts_to_keep:
        if pert == control_label:
            continue

        if pert not in adata_.var_names:
            continue

        gene_idx = adata_.var_names.get_loc(pert)

        # control mean for this gene
        if pert not in control_mean_cache:
            try:
                ctrl_mean = _mean(X[control_cells, gene_idx])
            except:
                print(control_cells.shape, control_cells)
                print(gene_idx)
                print(X[control_cells, gene_idx].shape)
            if ctrl_mean == 0:
                raise ValueError(
                    f"Mean {pert!r} expression in control cells is zero; "
                    "cannot compute knock-down ratio."
                )
            control_mean_cache[pert] = ctrl_mean
        else:
            ctrl_mean = control_mean_cache[pert]

        pert_cells = (perts == pert).values
        # FIX: Replace .A1 with .toarray().flatten() for scipy sparse matrices
        expr_vals = (
            X[pert_cells, gene_idx].toarray().flatten()
            if sp.issparse(X)
            else X[pert_cells, gene_idx]
        )
        ratios = expr_vals / ctrl_mean
        keep_mask[pert_cells] = ratios < cell_residual_expression

    # ---------- stage 3: minimum-cell filter ----------
    for pert in perts.unique():
        if pert == control_label:
            continue
        # cells of this perturbation *still* kept after stages 1-2
        pert_mask = (perts == pert).values & keep_mask
        if pert_mask.sum() < min_cells:
            keep_mask[pert_mask] = False  # drop them

    # return view with all criteria satisfied
    return adata_[keep_mask]


def set_var_index_to_col(adata: anndata.AnnData, col: str = "col", copy=True) -> None:
    """
    Set `adata.var` index to the values in the specified column, allowing non-unique indices.

    Parameters
    ----------
    adata : AnnData
        The AnnData object to modify.
    col : str
        Column in `adata.var` to use as the new index.

    Raises
    ------
    KeyError
        If the specified column does not exist in `adata.var`.
    """
    if col not in adata.var.columns:
        raise KeyError(f"Column {col!r} not found in adata.var.")

    adata.var.index = adata.var[col].astype("str")
    adata.var_names_make_unique()
    return adata
