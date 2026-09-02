import os
import h5py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, List, Optional, Tuple, Union
from pathlib import Path
import tqdm
import json
import pickle
import warnings
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MeshH5Dataset(Dataset):
    
    
    def __init__(self, h5_path: Union[str, Path]):
        self.h5_path = h5_path
        self.internal_fields = ['p', 'U', 'k', 'omega', 'nut', 'gammaInt', 'ReThetat'] 
        self.boundary_fields = ['Cp', 'Cf']
        self.constants = ['Cd', 'Cl', 'log_Re']

        with h5py.File(self.h5_path, 'r') as f:
            self.n = len(f.keys())
            self.case_names = f.keys()


        print(f"{self.n} cases loaded from {self.h5_path}")

    def __len__(self) -> int:
        return self.n


    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        key = self.case_names[idx]
        data = {}

        with h5py.File(self.h5_path, 'r') as f:
            case = f[key]


            internal_data = case['internal']

            data['internal_coords'] = torch.tensor(internal_data['coords'][:], dtype=torch.float32)
            data['sdf'] = torch.tensor(internal_data['sdf'][:], dtype=torch.float32).unsqueeze(-1)

            for field in self.internal_fields:
                data[field] = internal_data[field]
                if field != 'U':
                    data[field] = data[field].unsqueeze(-1)

            boundary_data = case['boundary']
            data['boundary_coords'] = torch.tensor(boundary_data['coords'][:], dtype=torch.float32)
            data['Cp'] = torch.tensor(boundary_data['Cp'], dtype=torch.float32).unsqueeze(-1)
            data['Cf'] = torch.tensor(boundary_data['Cf'], dtype=torch.float32).unsqueeze(-1)

            coeffs_data = case['constant']
            data['Cd'] = torch.tensor(coeffs_data['Cd'], dtype=torch.float32).unsqueeze(0)
            data['Cl'] = torch.tensor(coeffs_data['Cl'], dtype=torch.float32).unsqueeze(0)
            data['log_Re'] = torch.tensor(coeffs_data['log_Re'], dtype=torch.float32).unsqueeze(0)


        return data 

class FieldNormalizer:
    
    def __init__(self, fit_dataset: Optional[MeshH5Dataset] = None, save_path: Union[str, Path] = 'data_stats'):
        self.stats = {}
        self.fitted = False
        self.field_shapes = {}
        self.save_path = save_path 
        if fit_dataset is not None:
            self.fit(fit_dataset)
    
    def fit(self, dataset: MeshH5Dataset, verbose: bool = True):
        
        # Get all fields that need normalization
        fields_to_normalize = ['internal_coords', 'sdf', 'boundary_coords', *dataset.internal_fields, *dataset.boundary_fields, *dataset.constants]
        
        # Collect data for each field
        field_data = {field: [] for field in fields_to_normalize}
        
        for idx in tqdm(range(len(dataset)), desc="Collecting data"):
            sample = dataset[idx]
            for field in fields_to_normalize:
                if field in sample and isinstance(sample[field], torch.Tensor):
                    # Store shape info
                    if field not in self.field_shapes:
                        self.field_shapes[field] = sample[field].shape[1:]
                    field_data[field].append(sample[field].numpy())
        
        # Compute robust statistics
        stats_summary = {}
        for field, data_list in field_data.items():
            if not data_list:
                warnings.warn(f"No data found for field '{field}'")
                continue
            
            # Concatenate all data
            data = np.concatenate(data_list, axis=0)
            
            # Compute robust statistics
            self.stats[f'{field}_median'] = np.median(data, axis=0, keepdims=True)
            self.stats[f'{field}_q25'] = np.percentile(data, 25, axis=0, keepdims=True)
            self.stats[f'{field}_q75'] = np.percentile(data, 75, axis=0, keepdims=True)
            self.stats[f'{field}_iqr'] = self.stats[f'{field}_q75'] - self.stats[f'{field}_q25'] + 1e-8
            
            # Store min/max for monitoring (NOT for clipping!)
            self.stats[f'{field}_min'] = data.min(axis=0, keepdims=True)
            self.stats[f'{field}_max'] = data.max(axis=0, keepdims=True)
            
            # Store full range for reference
            self.stats[f'{field}_range'] = self.stats[f'{field}_max'] - self.stats[f'{field}_min']
            
            # Summary for verbose output
            stats_summary[field] = {
                'shape': data.shape,
                'median': np.median(data),
                'iqr': np.percentile(data, 75) - np.percentile(data, 25),
                'min': data.min(),
                'max': data.max(),
                'q25': np.percentile(data, 25),
                'q75': np.percentile(data, 75),
                'mean': data.mean(),
                'std': data.std()
            }
        
        self.fitted = True
        
        if verbose:
            print("\nNormalization Statistics Summary:")
            print("-" * 70)
            for field, stats in stats_summary.items():
                print(f"\n{field}:")
                print(f"  Shape: {stats['shape']}")
                print(f"  Median: {stats['median']:.4f}")
                print(f"  IQR:    {stats['iqr']:.4f}")
                print(f"  Min:    {stats['min']:.4f}")
                print(f"  Max:    {stats['max']:.4f}")
                print(f"  Q25:    {stats['q25']:.4f}")
                print(f"  Q75:    {stats['q75']:.4f}")
                if stats['iqr'] > 0:
                    print(f"  Range/ IQR: {stats['max'] - stats['min']:.2f} / {stats['iqr']:.2f}")
        
        print(f"   Fields normalized: {len(field_data)}")
        return self
    
    def normalize(self, data: torch.Tensor, field: str) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Normalizer not fitted yet. Call fit() first.")
        
        if field not in self.stats:
            return data
        
        if isinstance(data, np.ndarray):
            data = torch.from_numpy(data).float()
        
        # Use robust normalization: (x - median) / IQR
        median = torch.tensor(self.stats[f'{field}_median'], device=data.device)
        iqr = torch.tensor(self.stats[f'{field}_iqr'], device=data.device)
        
        normalized = (data - median) / iqr
        
        # NO CLIPPING! Preserve all values
        return normalized
    
    def denormalize(self, data: torch.Tensor, field: str) -> torch.Tensor:
        if not self.fitted:
            raise ValueError("Normalizer not fitted yet.")
        
        if field not in self.stats:
            return data
        
        median = torch.tensor(self.stats[f'{field}_median'], device=data.device)
        iqr = torch.tensor(self.stats[f'{field}_iqr'], device=data.device)
        
        return data * iqr + median
    
    def save(self):
        path = Path(self.save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as JSON for readability
        json_path = path.with_suffix('.json')
        stats_to_save = {}
        for key, value in self.stats.items():
            if isinstance(value, np.ndarray):
                stats_to_save[key] = value.tolist()
            else:
                stats_to_save[key] = value
        
        with open(json_path, 'w') as f:
            json.dump({
                'stats': stats_to_save,
                'fitted': self.fitted,
                'field_shapes': self.field_shapes
            }, f, indent=2)
        
        # Also save as pickle for faster loading
        with open(path, 'wb') as f:
            pickle.dump({
                'stats': self.stats,
                'fitted': self.fitted,
                'field_shapes': self.field_shapes
            }, f)
        
        print(f"Normalization stats saved to:")
        print(f"   Pickle: {path}")
        print(f"   JSON:   {json_path}")
        return self
    
    def load(self):
        path = Path(self.save_path)
        
        # Try pickle first
        if path.exists():
            with open(path, 'rb') as f:
                saved = pickle.load(f)
            self.stats = saved['stats']
            self.fitted = saved['fitted']
            self.field_shapes = saved.get('field_shapes', {})
        else:
            # Try JSON
            json_path = path.with_suffix('.json')
            if json_path.exists():
                with open(json_path, 'r') as f:
                    saved = json.load(f)
                # Convert lists back to numpy arrays
                self.stats = {}
                for key, value in saved['stats'].items():
                    if isinstance(value, list):
                        self.stats[key] = np.array(value)
                    else:
                        self.stats[key] = value
                self.fitted = saved['fitted']
                self.field_shapes = saved.get('field_shapes', {})
            else:
                raise FileNotFoundError(f"No normalization file found at {path} or {json_path}")
        
        print(f"Normalization stats loaded from {path}")
        print(f"   Method: Robust (Median/IQR)")
        return self
    
    def get_stats_summary(self, field: str) -> Dict:
        if not self.fitted:
            return {}
        
        if field not in self.stats:
            return {}
        
        return {
            'median': self.stats[f'{field}_median'],
            'iqr': self.stats[f'{field}_iqr'],
            'q25': self.stats[f'{field}_q25'],
            'q75': self.stats[f'{field}_q75'],
            'min': self.stats[f'{field}_min'],
            'max': self.stats[f'{field}_max'],
            'range': self.stats[f'{field}_range'],
            'shape': self.field_shapes.get(field, 'unknown')
        }

class normalizedMeshH5Dataset(MeshH5Dataset):

    def __init__(self, h5_path: Union[str, Path], save_path: Union[str, Path]="data_stats", load: bool=False):
        super().__init__(h5_path=h5_path)

        self.h5_path = h5_path
        self.save_path = save_path
        self.load = load

        self.normalizer=FieldNormalizer(save_path=save_path)


        if self.load:
            self.normalizer.load()
        else:
            self.normalizer.fit(dataset=MeshH5Dataset(h5_path=self.h5_path))
            self.normalizer.save()

        def __len__(self) -> int:
            return super().__len__()


        def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:

            raw = super().__getitem__(idx)

            normalized = {}
            for key, tensor in raw.items():
                normalized[key] = self.normalizer.normalize(tensor, key)

            return normalized


        def denormalize(self, item: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:

            raw = {}
            for key, tensor in item.items():
                raw[key] = self.normalizer.denormalize(tensor, key)

            return raw


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
            if 'constant' in f[case_name]:
                coeff_grp = f[case_name]['constant']
                print(f"  └── constant/")
                for key in coeff_grp.keys():
                    value = coeff_grp[key][()]
                    print(f"        ├── {key}: {value}")
            
            print()

if __name__ == "__main__":
    quick_inspect(Path("../sample_h5.h5"))
