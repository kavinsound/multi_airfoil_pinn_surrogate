import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeshH5Dataset(Dataset):
    
    
    def __init__(
        self,
        file_path: Union[str, Path],
        fields: Optional[List[str]] = None,
        include_coords: bool = True,
        include_sdf: bool = True,
        include_edges: bool = True,
        include_boundary: bool = False,
        include_coeffs: bool = False,
        normalize_coords: bool = True,
        normalize_fields: bool = False,
        field_stats: Optional[Dict] = None,
        transform: Optional[callable] = None,
        target_transform: Optional[callable] = None,
        cache_data: bool = False,
    ):
        
        self.file_path = Path(file_path)
        self.fields = fields or ['p', 'U', 'ReThetat', 'gammaInt', 'k', 'nut', 'omega']
        self.include_coords = include_coords
        self.include_sdf = include_sdf
        self.include_edges = include_edges
        self.include_boundary = include_boundary
        self.include_coeffs = include_coeffs
        self.normalize_coords = normalize_coords
        self.normalize_fields = normalize_fields
        self.field_stats = field_stats
        self.transform = transform
        self.target_transform = target_transform
        self.cache_data = cache_data
        
        # Validate file exists
        if not self.file_path.exists():
            raise FileNotFoundError(f"HDF5 file not found: {self.file_path}")
        
        # Get case names
        with h5py.File(self.file_path, 'r') as f:
            self.case_names = list(f.keys())
            self.num_cases = len(self.case_names)
        
        if self.num_cases == 0:
            raise ValueError(f"No cases found in HDF5 file: {self.file_path}")
        
        logger.info(f"Loaded dataset with {self.num_cases} cases")
        logger.info(f"Fields to load: {self.fields}")
        
        # Cache data if requested
        self.cached_data = {}
        if self.cache_data:
            logger.info("Caching all data in memory...")
            for case_name in self.case_names:
                self.cached_data[case_name] = self._load_case(case_name)
            logger.info("Caching complete")
        
        # Precompute normalization stats if needed
        if self.normalize_coords:
            self._compute_coord_stats()
        
        if self.normalize_fields and self.field_stats is None:
            self._compute_field_stats()
    
    def _compute_coord_stats(self):
        all_coords = []
        for case_name in self.case_names:
            with h5py.File(self.file_path, 'r') as f:
                coords = f[f'{case_name}/internal/coords'][:]
                all_coords.append(coords)
        
        all_coords = np.vstack(all_coords)
        self.coord_min = all_coords.min(axis=0)
        self.coord_max = all_coords.max(axis=0)
        self.coord_range = self.coord_max - self.coord_min
        self.coord_range[self.coord_range == 0] = 1.0  # Avoid division by zero
        
        logger.info(f"Coord stats - min: {self.coord_min}, max: {self.coord_max}")
    
    def _compute_field_stats(self):
        self.field_stats = {}
        
        for field in self.fields:
            all_values = []
            for case_name in self.case_names:
                with h5py.File(self.file_path, 'r') as f:
                    data = f[f'{case_name}/internal/{field}'][:]
                    all_values.append(data)
            
            all_values = np.vstack(all_values)
            self.field_stats[field] = {
                'mean': np.mean(all_values, axis=0),
                'std': np.std(all_values, axis=0)
            }
            # Avoid division by zero
            self.field_stats[field]['std'][self.field_stats[field]['std'] == 0] = 1.0
            
            logger.info(f"Field {field} stats - mean: {self.field_stats[field]['mean']}, "
                       f"std: {self.field_stats[field]['std']}")
    
    def _load_case(self, case_name: str) -> Dict:
        
        data = {}
        
        with h5py.File(self.file_path, 'r') as f:
            case_grp = f[case_name]
            internal_grp = case_grp['internal']
            
            # Load coordinates
            if self.include_coords:
                coords = internal_grp['coords'][:].astype(np.float32)
                data['coords'] = coords
                
                # Normalize if requested
                if self.normalize_coords:
                    data['coords'] = (coords - self.coord_min) / self.coord_range
            
            # Load SDF
            if self.include_sdf:
                sdf = internal_grp['sdf'][:].astype(np.float32)
                data['sdf'] = sdf
            
            # Load fields
            for field in self.fields:
                if field in internal_grp:
                    field_data = internal_grp[field][:].astype(np.float32)
                    
                    # Normalize if requested
                    if self.normalize_fields and self.field_stats is not None:
                        if field in self.field_stats:
                            field_data = (field_data - self.field_stats[field]['mean']) / \
                                        self.field_stats[field]['std']
                    
                    data[field] = field_data
                else:
                    logger.warning(f"Field {field} not found in case {case_name}")
                    # Create empty array as placeholder
                    n_points = internal_grp['coords'].shape[0]
                    data[field] = np.zeros((n_points,), dtype=np.float32)
            
            # Load edges (connectivity from Delaunay triangulation)
            if self.include_edges:
                if 'edges' in internal_grp:
                    data['edges'] = internal_grp['edges'][:].astype(np.int32)
                else:
                    logger.warning(f"No edges found in case {case_name}")
                    data['edges'] = np.array([], dtype=np.int32)
            
            # Load boundary data
            if self.include_boundary:
                boundary_grp = case_grp['boundary']
                boundary_coords = boundary_grp['coords'][:].astype(np.float32)
                data['boundary_coords'] = boundary_coords
                
                for key in boundary_grp.keys():
                    if key != 'coords':
                        data[f'boundary_{key}'] = boundary_grp[key][:].astype(np.float32)
            
            # Load coefficients
            if self.include_coeffs:
                coeff_grp = case_grp['coeffs']
                for key in coeff_grp.keys():
                    data[key] = coeff_grp[key][()].astype(np.float32)
        
        return data
    
    def __len__(self) -> int:
        return self.num_cases
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        
        case_name = self.case_names[idx]
        
        # Load from cache or directly from file
        if self.cache_data:
            data = self.cached_data[case_name]
        else:
            data = self._load_case(case_name)
        
        # Convert to tensors
        tensor_data = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                if value.dtype == np.int32:
                    tensor_data[key] = torch.tensor(value, dtype=torch.long)
                else:
                    tensor_data[key] = torch.tensor(value, dtype=torch.float32)
            else:
                tensor_data[key] = torch.tensor(value, dtype=torch.float32)
        
        # Apply transforms if provided
        if self.transform is not None:
            # Transform input data (coordinates, SDF, fields)
            input_data = {k: v for k, v in tensor_data.items() 
                         if k in ['coords', 'sdf', 'edges'] + self.fields}
            tensor_data.update(self.transform(input_data))
        
        if self.target_transform is not None:
            # Transform target data (coefficients)
            target_data = {k: v for k, v in tensor_data.items() 
                          if k in ['Cd', 'Cl']}
            tensor_data.update(self.target_transform(target_data))
        
        return tensor_data


class MeshDataLoader:
    
    
    @staticmethod
    def create_dataloader(
        file_path: Union[str, Path],
        batch_size: int = 32,
        shuffle: bool = True,
        num_workers: int = 4,
        pin_memory: bool = True,
        drop_last: bool = False,
        fields: Optional[List[str]] = None,
        include_coords: bool = True,
        include_sdf: bool = True,
        include_edges: bool = True,
        include_boundary: bool = False,
        include_coeffs: bool = False,
        normalize_coords: bool = True,
        normalize_fields: bool = False,
        cache_data: bool = False,
        **kwargs
    ) -> DataLoader:
        
        dataset = MeshH5Dataset(
            file_path=file_path,
            fields=fields,
            include_coords=include_coords,
            include_sdf=include_sdf,
            include_edges=include_edges,
            include_boundary=include_boundary,
            include_coeffs=include_coeffs,
            normalize_coords=normalize_coords,
            normalize_fields=normalize_fields,
            cache_data=cache_data
        )
        
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
            **kwargs
        )
        
        return dataloader


def create_dataset(
    h5_file_path: Union[str, Path],
    fields: Optional[List[str]] = None,
    include_coords: bool = True,
    include_sdf: bool = True,
    include_edges: bool = True,
    include_boundary: bool = True,
    include_coeffs: bool = True,
    normalize_coords: bool = True,
    normalize_fields: bool = True,
    cache_data: bool = False,
    **kwargs
) -> MeshH5Dataset:
    
    return MeshH5Dataset(
        file_path=h5_file_path,
        fields=fields,
        include_coords=include_coords,
        include_sdf=include_sdf,
        include_edges=include_edges,
        include_boundary=include_boundary,
        include_coeffs=include_coeffs,
        normalize_coords=normalize_coords,
        normalize_fields=normalize_fields,
        cache_data=cache_data,
        **kwargs
    )


def create_dataloader(
    h5_file_path: Union[str, Path],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 4,
    pin_memory: bool = True,
    drop_last: bool = False,
    fields: Optional[List[str]] = None,
    include_coords: bool = True,
    include_sdf: bool = True,
    include_edges: bool = True,
    include_boundary: bool = True,
    include_coeffs: bool = True,
    normalize_coords: bool = True,
    normalize_fields: bool = True,
    cache_data: bool = False,
    **kwargs
) -> DataLoader:
    
    dataset = create_dataset(
        h5_file_path=h5_file_path,
        fields=fields,
        include_coords=include_coords,
        include_sdf=include_sdf,
        include_edges=include_edges,
        include_boundary=include_boundary,
        include_coeffs=include_coeffs,
        normalize_coords=normalize_coords,
        normalize_fields=normalize_fields,
        cache_data=cache_data
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
        **kwargs
    )

def create_train_val_test_loaders(
    h5_file_path: Union[str, Path],
    batch_size: int = 32,
    train_split: float = 0.8,
    val_split: float = 0.1,
    test_split: float = 0.1,
    random_seed: int = 42,
    num_workers: int = 4,
    pin_memory: bool = True,
    fields: Optional[List[str]] = None,
    include_coords: bool = True,
    include_sdf: bool = True,
    include_edges: bool = True,
    include_boundary: bool = True,
    include_coeffs: bool = True,
    normalize_coords: bool = True,
    normalize_fields: bool = True,
    cache_data: bool = False,
    **kwargs
) -> Dict[str, DataLoader]:
    
    dataset = create_dataset(
        h5_file_path=h5_file_path,
        fields=fields,
        include_coords=include_coords,
        include_sdf=include_sdf,
        include_edges=include_edges,
        include_boundary=include_boundary,
        include_coeffs=include_coeffs,
        normalize_coords=normalize_coords,
        normalize_fields=normalize_fields,
        cache_data=cache_data
    )
    
    case_names = dataset.case_names
    
    np.random.seed(random_seed)
    indices = np.random.permutation(len(case_names))
    
    train_end = int(train_split * len(indices))
    val_end = int((train_split + val_split) * len(indices))
    
    train_indices = indices[:train_end]
    val_indices = indices[train_end:val_end]
    test_indices = indices[val_end:]
    
    from torch.utils.data import Subset
    
    train_dataset = Subset(dataset, train_indices)
    val_dataset = Subset(dataset, val_indices)
    test_dataset = Subset(dataset, test_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
        **kwargs
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        **kwargs
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
        **kwargs
    )
    
    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader
    }


def quick_inspect(h5_file_path: Union[str, Path]) -> None:
   
    file_path = Path(h5_file_path)
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    print(f"\n{'='*70}")
    print(f"📁 HDF5 File: {file_path.name}")
    print(f"{'='*70}")
    
    with h5py.File(file_path, 'r') as f:
        cases = list(f.keys())
        print(f"Total cases: {len(cases)}\n")
        
        for case_name in cases:
            print(f"📂 Case: {case_name}")
            
            # Internal group
            if 'internal' in f[case_name]:
                internal_grp = f[case_name]['internal']
                print(f"  ├── internal/")
                for key in internal_grp.keys():
                    shape = internal_grp[key].shape
                    dtype = internal_grp[key].dtype
                    print(f"  │     ├── {key}: {shape}, {dtype}")
            
            # Boundary group
            if 'boundary' in f[case_name]:
                boundary_grp = f[case_name]['boundary']
                print(f"  ├── boundary/")
                for key in boundary_grp.keys():
                    shape = boundary_grp[key].shape
                    dtype = boundary_grp[key].dtype
                    print(f"  │     ├── {key}: {shape}, {dtype}")
            
            # Coefficients group
            if 'coeffs' in f[case_name]:
                coeff_grp = f[case_name]['coeffs']
                print(f"  └── coeffs/")
                for key in coeff_grp.keys():
                    value = coeff_grp[key][()]
                    print(f"        ├── {key}: {value}")
            
            print()

if __name__ == "__main__":
    quick_inspect(Path("../sample_h5.h5"))
