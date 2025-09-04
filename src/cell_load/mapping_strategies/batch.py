import logging
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from ..dataset import PerturbationDataset

from .mapping_strategies import BaseMappingStrategy

# New Memory Mapping Imports (Dan)
import numpy as np
from typing import Iterator, List, Tuple, Optional, Union, Any
from pathlib import Path
from ..utils.data_utils import MemoryMappedArray, create_memory_mapped_dataset

logger = logging.getLogger(__name__)

# ============================================================================
# MEMORY-MAPPED BATCH STRATEGIES (Dan)
# ============================================================================

class MemoryMappedBatchStrategy:
    """
    Base class for Memory-mapped Batch strategies
    Extends your existing Batch strategies
    """
    def __init__(self, 
                 batch_size: int = 32,
                 cache_dir: Optional[str] = None,
                 preload_batches: bool = True):
        """
        Args:
            batch_size: Batch size
            cache_dir: Directory for memory-mapped cache files
            preload_batches: Preload batches into memory-mapped files
        """
        self.batch_size = batch_size
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.preload_batches = preload_batches
        self.batch_cache = {}  # Cache for memory-mapped batches
        
        if self.cache_dir:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            
    def _create_batch_cache_key(self, data_id: str, batch_idx: int) -> str:
        """Create cache key for batch"""
        return f"batch_{data_id}_{batch_idx}_{self.batch_size}"
        
    def _get_cached_batch(self, cache_key: str) -> Optional[MemoryMappedArray]:
        """Load cached batch"""
        if cache_key in self.batch_cache:
            return self.batch_cache[cache_key]
        return None
        
    def _cache_batch(self, 
                    batch_data: np.ndarray, 
                    cache_key: str) -> MemoryMappedArray:
        """Store batch in memory-mapped cache"""
        if self.cache_dir is None:
            import tempfile
            self.cache_dir = Path(tempfile.mkdtemp(prefix="cell_load_batch_cache_"))
            
        cache_file = self.cache_dir / f"{cache_key}.mmap"
        mmap_batch = create_memory_mapped_dataset(batch_data, cache_file, overwrite=True)
        
        self.batch_cache[cache_key] = mmap_batch
        return mmap_batch
        
    def cleanup_cache(self):
        """Clean up memory-mapped cache"""
        for mmap_array in self.batch_cache.values():
            mmap_array.close()
        self.batch_cache.clear()
        
        if self.cache_dir and self.cache_dir.exists():
            import shutil
            shutil.rmtree(self.cache_dir)
            
    def __del__(self):
        self.cleanup_cache()

class MemoryMappedRandomBatchStrategy(MemoryMappedBatchStrategy):
    """
    Random Batching with Memory Mapping
    Optimized for large datasets with random access
    """
    def __init__(self, *args, seed: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)
            
    def create_batches(self, 
                      data: Union[np.ndarray, MemoryMappedArray],
                      data_id: str = "default") -> Iterator[MemoryMappedArray]:
        """
        Create random batches with memory mapping
        
        Args:
            data: Input data (NumPy array or MemoryMappedArray)
            data_id: Identifier for caching
            
        Yields:
            Memory-mapped batch arrays
        """
        n_samples = len(data)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        # Create random permutation
        indices = np.random.permutation(n_samples)
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_samples)
            batch_indices = indices[start_idx:end_idx]
            
            cache_key = self._create_batch_cache_key(data_id, batch_idx)
            
            # Check cache first
            cached_batch = self._get_cached_batch(cache_key)
            if cached_batch is not None:
                yield cached_batch
                continue
                
            # Create new batch
            batch_data = data[batch_indices]
            
            # Convert to numpy if needed
            if not isinstance(batch_data, np.ndarray):
                batch_data = np.array(batch_data)
                
            # Cache batch if enabled
            if self.preload_batches:
                mmap_batch = self._cache_batch(batch_data, cache_key)
                yield mmap_batch
            else:
                # Return as temporary memory-mapped array
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mmap')
                temp_file.close()
                
                temp_mmap = create_memory_mapped_dataset(
                    batch_data, temp_file.name, overwrite=True
                )
                yield temp_mmap

class MemoryMappedSequentialBatchStrategy(MemoryMappedBatchStrategy):
    """
    Sequential Batching with Memory Mapping
    Optimized for sequential access patterns
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def create_batches(self, 
                      data: Union[np.ndarray, MemoryMappedArray],
                      data_id: str = "default") -> Iterator[MemoryMappedArray]:
        """
        Create sequential batches with memory mapping
        
        Args:
            data: Input data (NumPy array or MemoryMappedArray)
            data_id: Identifier for caching
            
        Yields:
            Memory-mapped batch arrays
        """
        n_samples = len(data)
        n_batches = (n_samples + self.batch_size - 1) // self.batch_size
        
        for batch_idx in range(n_batches):
            start_idx = batch_idx * self.batch_size
            end_idx = min(start_idx + self.batch_size, n_samples)
            
            cache_key = self._create_batch_cache_key(data_id, batch_idx)
            
            # Check cache first
            cached_batch = self._get_cached_batch(cache_key)
            if cached_batch is not None:
                yield cached_batch
                continue
                
            # Create new batch
            batch_data = data[start_idx:end_idx]
            
            # Convert to numpy if needed
            if not isinstance(batch_data, np.ndarray):
                batch_data = np.array(batch_data)
                
            # Cache batch if enabled
            if self.preload_batches:
                mmap_batch = self._cache_batch(batch_data, cache_key)
                yield mmap_batch
            else:
                # Return as temporary memory-mapped array
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mmap')
                temp_file.close()
                
                temp_mmap = create_memory_mapped_dataset(
                    batch_data, temp_file.name, overwrite=True
                )
                yield temp_mmap

class MemoryMappedStratifiedBatchStrategy(MemoryMappedBatchStrategy):
    """
    Stratified Batching with Memory Mapping
    Ensures balanced representation across batches
    """
    def __init__(self, *args, stratify_column: Optional[int] = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.stratify_column = stratify_column
        
    def create_batches(self, 
                      data: Union[np.ndarray, MemoryMappedArray],
                      labels: Optional[np.ndarray] = None,
                      data_id: str = "default") -> Iterator[MemoryMappedArray]:
        """
        Create stratified batches with memory mapping
        
        Args:
            data: Input data (NumPy array or MemoryMappedArray)
            labels: Labels for stratification
            data_id: Identifier for caching
            
        Yields:
            Memory-mapped batch arrays
        """
        n_samples = len(data)
        
        if labels is None and self.stratify_column is not None:
            # Use specified column from data for stratification
            labels = data[:, self.stratify_column]
        elif labels is None:
            # Fallback to sequential batching
            yield from MemoryMappedSequentialBatchStrategy.create_batches(
                self, data, data_id
            )
            return
            
        # Get unique labels and their indices
        unique_labels = np.unique(labels)
        label_indices = {label: np.where(labels == label)[0] for label in unique_labels}
        
        # Calculate samples per label per batch
        samples_per_label = self.batch_size // len(unique_labels)
        remainder = self.batch_size % len(unique_labels)
        
        batch_idx = 0
        label_positions = {label: 0 for label in unique_labels}
        
        while any(pos < len(indices) for pos, indices in 
                 zip(label_positions.values(), label_indices.values())):
            
            batch_indices = []
            
            # Sample from each label
            for i, label in enumerate(unique_labels):
                indices = label_indices[label]
                pos = label_positions[label]
                
                # Add extra sample for remainder
                n_samples_this_label = samples_per_label + (1 if i < remainder else 0)
                
                end_pos = min(pos + n_samples_this_label, len(indices))
                batch_indices.extend(indices[pos:end_pos])
                label_positions[label] = end_pos
                
            if not batch_indices:
                break
                
            # Shuffle batch indices
            np.random.shuffle(batch_indices)
            
            cache_key = self._create_batch_cache_key(data_id, batch_idx)
            
            # Check cache first
            cached_batch = self._get_cached_batch(cache_key)
            if cached_batch is not None:
                yield cached_batch
                batch_idx += 1
                continue
                
            # Create new batch
            batch_data = data[batch_indices]
            
            # Convert to numpy if needed
            if not isinstance(batch_data, np.ndarray):
                batch_data = np.array(batch_data)
                
            # Cache batch if enabled
            if self.preload_batches:
                mmap_batch = self._cache_batch(batch_data, cache_key)
                yield mmap_batch
            else:
                # Return as temporary memory-mapped array
                import tempfile
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mmap')
                temp_file.close()
                
                temp_mmap = create_memory_mapped_dataset(
                    batch_data, temp_file.name, overwrite=True
                )
                yield temp_mmap
                
            batch_idx += 1

# ============================================================================
# ADAPTIVE BATCH STRATEGY - Automatically chooses best strategy (Dan)
# ============================================================================

class AdaptiveMemoryMappedBatchStrategy:
    """
    Adaptive Batch Strategy that automatically chooses the best approach
    Based on data characteristics and access patterns
    """
    def __init__(self, 
                 batch_size: int = 32,
                 cache_dir: Optional[str] = None,
                 auto_optimize: bool = True):
        """
        Args:
            batch_size: Batch size
            cache_dir: Directory for memory-mapped cache files
            auto_optimize: Automatically optimize strategy based on usage
        """
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.auto_optimize = auto_optimize
        
        # Available strategies
        self.strategies = {
            'sequential': MemoryMappedSequentialBatchStrategy(
                batch_size, cache_dir, preload_batches=True
            ),
            'random': MemoryMappedRandomBatchStrategy(
                batch_size, cache_dir, preload_batches=True
            ),
            'stratified': MemoryMappedStratifiedBatchStrategy(
                batch_size, cache_dir, preload_batches=True
            )
        }
        
        # Performance tracking
        self.strategy_performance = {}
        self.current_strategy = 'sequential'
        
    def _analyze_data_characteristics(self, data: Union[np.ndarray, MemoryMappedArray]) -> str:
        """
        Analyze data characteristics to choose optimal strategy
        
        Args:
            data: Input data
            
        Returns:
            Recommended strategy name
        """
        n_samples, n_features = data.shape if len(data.shape) > 1 else (len(data), 1)
        
        # For small datasets, use sequential
        if n_samples < 1000:
            return 'sequential'
            
        # For very large datasets, use random to avoid memory issues
        if n_samples > 100000:
            return 'random'
            
        # For medium datasets, use sequential by default
        return 'sequential'
        
    def create_batches(self, 
                      data: Union[np.ndarray, MemoryMappedArray],
                      labels: Optional[np.ndarray] = None,
                      data_id: str = "default",
                      strategy: Optional[str] = None) -> Iterator[MemoryMappedArray]:
        """
        Create batches using adaptive strategy
        
        Args:
            data: Input data
            labels: Labels for stratification (optional)
            data_id: Identifier for caching
            strategy: Force specific strategy (optional)
            
        Yields:
            Memory-mapped batch arrays
        """
        # Choose strategy
        if strategy is None:
            if self.auto_optimize:
                strategy = self._analyze_data_characteristics(data)
            else:
                strategy = self.current_strategy
                
        # Use appropriate strategy
        if strategy == 'stratified' and labels is not None:
            batch_generator = self.strategies['stratified'].create_batches(
                data, labels, data_id
            )
        else:
            batch_generator = self.strategies[strategy].create_batches(data, data_id)
            
        # Track performance if auto-optimization is enabled
        if self.auto_optimize:
            import time
            start_time = time.time()
            batch_count = 0
            
            for batch in batch_generator:
                yield batch
                batch_count += 1
                
            # Record performance
            total_time = time.time() - start_time
            self.strategy_performance[strategy] = {
                'total_time': total_time,
                'avg_time_per_batch': total_time / max(batch_count, 1),
                'batch_count': batch_count
            }
            
            # Update current strategy if this one performed better
            self._update_optimal_strategy()
        else:
            yield from batch_generator
            
    def _update_optimal_strategy(self):
        """Update optimal strategy based on performance history"""
        if len(self.strategy_performance) < 2:
            return
            
        # Find strategy with best average time per batch
        best_strategy = min(
            self.strategy_performance.items(),
            key=lambda x: x[1]['avg_time_per_batch']
        )[0]
        
        if best_strategy != self.current_strategy:
            print(f"🔄 Switching to {best_strategy} strategy (better performance)")
            self.current_strategy = best_strategy
            
    def get_performance_stats(self) -> dict:
        """Get performance statistics for all strategies"""
        return self.strategy_performance.copy()
        
    def cleanup_cache(self):
        """Clean up all strategy caches"""
        for strategy in self.strategies.values():
            strategy.cleanup_cache()

# ============================================================================
# INTEGRATION HELPERS - For existing batch.py (Dan)
# ============================================================================

def enhance_existing_batch_strategy(existing_strategy_class):
    """
    Decorator to enhance existing batch strategies with memory mapping
    
    Usage:
        @enhance_existing_batch_strategy
        class YourExistingBatchStrategy:
            ...
    """
    class EnhancedBatchStrategy(MemoryMappedBatchStrategy, existing_strategy_class):
        def __init__(self, *args, **kwargs):
            # Extract memory mapping parameters
            mmap_params = {
                'cache_dir': kwargs.pop('cache_dir', None),
                'preload_batches': kwargs.pop('preload_batches', True),
            }
            
            # Initialize both parent classes
            MemoryMappedBatchStrategy.__init__(self, **mmap_params)
            existing_strategy_class.__init__(self, *args, **kwargs)
            
    return EnhancedBatchStrategy

# ============================================================================
# FACTORY FUNCTIONS - Easy creation (Dan)
# ============================================================================

def create_memory_mapped_batch_strategy(strategy_type: str = 'adaptive',
                                       batch_size: int = 32,
                                       cache_dir: Optional[str] = None,
                                       **kwargs):
    """
    Factory function to create memory-mapped batch strategies
    
    Args:
        strategy_type: Type of strategy ('sequential', 'random', 'stratified', 'adaptive')
        batch_size: Batch size
        cache_dir: Cache directory
        **kwargs: Additional parameters
        
    Returns:
        Memory-mapped batch strategy instance
    """
    if strategy_type == 'sequential':
        return MemoryMappedSequentialBatchStrategy(batch_size, cache_dir, **kwargs)
    elif strategy_type == 'random':
        return MemoryMappedRandomBatchStrategy(batch_size, cache_dir, **kwargs)
    elif strategy_type == 'stratified':
        return MemoryMappedStratifiedBatchStrategy(batch_size, cache_dir, **kwargs)
    elif strategy_type == 'adaptive':
        return AdaptiveMemoryMappedBatchStrategy(batch_size, cache_dir, **kwargs)
    else:
        raise ValueError(f"Unknown strategy type: {strategy_type}")

class BatchMappingStrategy(BaseMappingStrategy):
    """
    Maps a perturbed cell to random control(s) drawn from the same batch and cell type.
    If no controls are available in the same batch, falls back to controls from the same cell type.

    This strategy matches the RandomMappingStrategy structure except it groups the control cells
    by the tuple (batch, cell_type) instead of just by cell type.
    """

    def __init__(self, name="batch", random_state=42, n_basal_samples=1, **kwargs):
        super().__init__(name, random_state, n_basal_samples, **kwargs)
        # For each split, store a mapping: {(batch, cell_type): [ctrl_indices]}
        self.split_control_maps = {
            "train": {},
            "train_eval": {},
            "val": {},
            "test": {},
        }

    def name():
        """Name of the mapping strategy."""
        return "batch"

    def register_split_indices(
        self,
        dataset: "PerturbationDataset",
        split: str,
        _perturbed_indices: np.ndarray,
        control_indices: np.ndarray,
    ):
        """
        Build a map from (batch, cell_type) to control indices for the given split.
        For each control cell, we retrieve both its batch and cell type, using that pair as the key.
        """
        for idx in control_indices:
            batch = dataset.get_batch(idx)
            cell_type = dataset.get_cell_type(idx)
            key = (batch, cell_type)
            if key not in self.split_control_maps[split]:
                self.split_control_maps[split][key] = []
            self.split_control_maps[split][key].append(idx)

    def get_control_indices(
        self, dataset: "PerturbationDataset", split: str, perturbed_idx: int
    ) -> np.ndarray:
        """
        Return n_basal_samples control indices for the perturbed cell that are
        from the same batch and the same cell type.

        If the batch group for the perturbed cell is empty, the method falls back to
        using all control cells from the same cell type (regardless of batch).
        """
        batch = dataset.get_batch(perturbed_idx)
        cell_type = dataset.get_cell_type(perturbed_idx)
        key = (batch, cell_type)
        pool = self.split_control_maps[split].get(key, [])

        if not pool:
            # Fallback: If no controls exist in this batch, select from all controls with the same cell type.
            pool = []
            for (b, ct), indices in self.split_control_maps[split].items():
                if ct == cell_type:
                    pool.extend(indices)

        if not pool:
            raise ValueError(
                "No control cells found in BatchMappingStrategy for cell type '{}'".format(
                    cell_type
                )
            )

        return self.rng.choice(pool, size=self.n_basal_samples, replace=True)

    def get_control_index(
        self, dataset: "PerturbationDataset", split: str, perturbed_idx: int
    ):
        """
        Returns a single control index for the perturbed cell.
        This method first attempts to select from controls in the same batch and cell type.
        If no controls are present in the same batch, it falls back to all controls from the same cell type.
        """
        batch = dataset.get_batch(perturbed_idx)
        cell_type = dataset.get_cell_type(perturbed_idx)
        key = (batch, cell_type)
        pool = self.split_control_maps[split].get(key, [])

        if not pool:
            # Fallback: select from controls that are of the same cell type regardless of batch.
            pool = []
            for (b, ct), indices in self.split_control_maps[split].items():
                if ct == cell_type:
                    pool.extend(indices)

        if not pool:
            return None

        return self.rng.choice(pool)
