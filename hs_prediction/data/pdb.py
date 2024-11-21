from __future__ import annotations
from typing import Union, List
import pandas as pd
import numpy as np
from numpy.typing import NDArray


class PDB:
    """Fast and simple PDB parser using pandas"""

    def __init__(self, dataframe: pd.DataFrame):
        self.df = dataframe

    @classmethod
    def from_file(cls, filepath: str) -> PDB:
        """Load PDB file"""

        data = [
            [
                line[:6].strip(), int(line[6:11]), line[12:16].strip(),
                line[17:20], line[21:22], int(line[22:26]),
                float(line[30:38]), float(line[38:46]), float(line[46:54]),
                float(line[54:60]), float(line[60:66]), line[76:78].strip()
            ]
            for line in open(filepath, 'r').read().split('\n')
            if line[:6].strip() in ['ATOM', 'HETATM']
        ]
        # Parsing manually with known types is much faster than pd.read_fwf
        df = pd.DataFrame(
            data,
            columns=['type', 'id', 'name', 'resn', 'chain', 'resi', 'x', 'y', 'z', 'occ', 'fac', 'sym']
        )

        return cls(df)

    @classmethod
    def from_coords(cls, coords: NDArray, resn: str = 'LIG') -> PDB:
        """Create PDB from coordinates"""
        df = pd.DataFrame(
            coords,
            columns=['x', 'y', 'z']
        )
        df['type'] = 'HETATM'
        df['id'] = np.arange(len(df)) + 1
        df['name'] = 'O'
        df['resn'] = resn
        df['chain'] = 'A'
        df['resi'] = np.arange(len(df)) + 1
        df['occ'] = 1.0
        df['fac'] = 1.0
        df['sym'] = 'O'
        return cls(df)

    @classmethod
    def from_pdb_list(cls, pdb_list: List[PDB]) -> PDB:
        df = []
        for i, pdb in enumerate(pdb_list):
            pdb.df['model'] = i + 1
            df.append(pdb.df)
        return cls(pd.concat(df))


    @staticmethod
    def _write_pdb_line(fo, row):
        fo.write(
            f"ATOM  {row['id']:5} {row['name']:4} {row['resn']:3} {row['chain']:1}{row['resi']:4}    "
            f"{row['x']:8.3f}{row['y']:8.3f}{row['z']:8.3f}{row['occ']:6.2f}{row['fac']:6.2f}          "
            f"{row['sym']:2}\n"
        )

    def save(self, filepath: str):
        """Save PDB file"""
        assert filepath.endswith('.pdb'), 'Filepath must end with .pdb'
        with open(filepath, 'w') as fo:
            if 'model' in self.df.columns:
                for model in self.unique('model'):
                    fo.write(f"MODEL {model}\n")
                    for _, row in self.df[self.df['model'] == model].iterrows():
                        self._write_pdb_line(fo, row)
                    fo.write("ENDMDL\n")
            else:
                for _, row in self.df.iterrows():
                    self._write_pdb_line(fo, row)

    def select(self, col: str, val: Union[str, int, float, bool]) -> PDB:
        """Select atoms based on column value"""
        return PDB(self.df[self.df[col] == val])

    def select_box(self, lower: NDArray, upper: NDArray) -> PDB:
        """Select atoms within a box"""
        coords = self.get_coords()
        mask = np.all(coords > lower, axis=1) & np.all(coords < upper, axis=1)
        return PDB(self[mask])

    def select_sphere(self, center, radius):
        """Select atoms within a sphere"""
        norms = np.linalg.norm(self.get_coords() - center, axis=1)
        return PDB(self[norms < radius])

    def remove_het(self) -> PDB:
        """Remove heteroatoms"""
        return PDB(self.df[self.df['type'] == 'ATOM'])

    def remove_water(self) -> PDB:
        """Remove water"""
        water_resids = ['HOH', 'WAT', 'SPC']
        return PDB(self.df[~self.df['resn'].isin(water_resids)])

    def remove_hydrogen(self) -> PDB:
        """Remove hydrogen"""
        return PDB(self.df[self.df['sym'] != 'H'])

    def set_coords(self, coords: NDArray) -> None:
        """Set coordinates of atoms"""
        self.df.loc[:, ['x', 'y', 'z']] = coords

    def get_coords(self):
        """Get coordinates of atoms"""
        return self.df[['x', 'y', 'z']].values.astype(float)

    def unique(self, key):
        """Get unique values of a column"""
        return np.sort(pd.unique(self.df[key]))

    def copy(self):
        """Copy PDB object"""
        return self.__copy__()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, key):
        return self.df[key]

    def __copy__(self):
        return PDB(self.df.copy())
