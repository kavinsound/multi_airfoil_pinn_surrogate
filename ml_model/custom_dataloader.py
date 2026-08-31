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
    
    
    def __init__(self, h5_path: Union[str, Path]):
        self.h5_path = h5_path
        self.internal_fields = ['p', 'U', 'k', 'omega', 'nut', 'gammaInt', 'ReThetat'] 
        self.boundary_fields = ['Cp', 'Cf']
        self.constants = ['Cd', 'Cl', 'Re']

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
            data['Re'] = torch.tensor(coeffs_data['Re'], dtype=torch.float32).unsqueeze(0)


        return data 



class MeshDataLoader:
    
    
    ...




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
