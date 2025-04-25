import os
from rdkit import RDLogger
from rdkit import Chem
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Required
from pathlib import Path
from augchem.modules.smiles.smiles_modules import *
# disable rdkit warnings
RDLogger.DisableLog('rdApp.*')

class Loader:
    def __init__(self, path: Path):
        self.path = path

    def load_qm9_xyz(self, file_path):
        """Load a single QM9.xyz file."""
        with open(file_path, 'r') as f:
            # Number of atoms
            natoms = int(f.readline())
            # Properties are in the second line
            properties = list(map(float, f.readline().split()[2:]))
            # Read atomic coordinates and types
            atoms = []
            coordinates = []
            smiles1 = ''
            smiles2 = '' 
            inchi1 = ''
            inchi2 = ''
            # print(properties)
            for num_line, line in enumerate(f):
                # print(num_line, line)
                if num_line >= 0 and num_line < natoms:
                    info = line.replace("*^","e").split()
                    atoms.append(info[0])
                    coordinates.append(list(map(float, info[1:-1])))
                if num_line == natoms + 1:
                    smiles1, smiles2 = line.split()
                if num_line == natoms + 2:
                    inchi1, inchi2 = line.split()
            
        return {
            "natoms": natoms,
            "atoms": atoms,
            "coordinates": np.array(coordinates),
            "smiles_1": smiles1,
            "smiles_2": smiles2,
            "inchi_1": inchi1,
            "inchi_2": inchi2,
            "properties": properties
        }

    def load_qm9_dataset(self, directory_path, list_mols=[]):
        """Load the entire QM9 dataset from a directory containing .xyz files."""
        X = []
        Y = []
        S = []
        SMILES1 = []
        SMILES2 = []
        INCHI1 = []
        INCHI2 = []

        for file_name in os.listdir(directory_path):
            if file_name.endswith(".xyz"):
                file_path = os.path.join(directory_path, file_name)
                molecule_data = self.load_qm9_xyz(file_path)
                if molecule_data['natoms'] in list_mols or len(list_mols)==0:
                    X.append([molecule_data['atoms'], molecule_data['coordinates']])
                    Y.append(molecule_data['properties'])
                    S.append(molecule_data['natoms'])
                    SMILES1.append(molecule_data['smiles_1'])
                    SMILES2.append(molecule_data['smiles_2'])
                    INCHI1.append(molecule_data['inchi_1'])
                    INCHI2.append(molecule_data['inchi_2'])

        return X, Y, S, SMILES1, SMILES2, INCHI1, INCHI2
    
    def qm9_to_csv(self, qm9_folder: Path, property: int):
        XYZ, Y, natoms, SMILES_1, SMILES_2, INCHI_1, INCHI_2 = self.load_qm9_dataset(qm9_folder)

        z = Y.tolist()

        all_properties = []

        for i, propriedades in enumerate(z):
            propriedade = propriedades[property]

            all_properties.append(propriedade)

        dataframe = pd.DataFrame(
            {
                'SMILES_1': SMILES_1,
                'SMILES_2': SMILES_2,
                'INCHI_1': INCHI_1,
                'INCHI_2': INCHI_2,
                'Property_0': all_properties
            }
        )

        dataframe.to_csv('QM9.csv', index=True, float_format='%.8e')

# estruturar esse projeto pra conseguir fazer o pip local
class Augmentator:
    def __init__(self, seed: int = 4123):
        self.seed = seed
        self.ss = np.random.SeedSequence(self.seed)
        self.rng = np.random.RandomState(seed=seed)

        self.SMILES = self.SMILESModule(self)
        self.Graphs = self.GraphsModule(self)
        self.INCHI = self.INCHIModule(self)

    class SMILESModule:
        def __init__(self, parent):
            self.parent = parent

        # Receber csv, colocar coluna extra de qual id veio o aumento, adicionar tag de qual metodo vai ser utilizado e aumento de %: inicial 1000 dados -> 1200 dados, essa é a %
        def augment_data(self, dataset: Path, mask_ratio: float = 0.1, delete_ratio: float = 0.3, seed: int = 42, 
                            augment_percentage: float = 0.2, augmentation_methods: List[str] = ["fusion", "enumerate"], col_to_augment: str = 'SMILES',
                            property_col: str = None) -> pd.DataFrame:
            """
            Augment SMILES strings using fusion and enumeration methods.
            
            # Parameters:
            `dataset`: Path - List of SMILES strings to augment

            `mask_ratio`: float - Ratio of tokens to mask in fusion method

            `delete_ratio`: float - Ratio of tokens to delete in fusion method

            `attempts`: int - Number of attempts for SMILES enumeration

            `max_unique`: int - Maximum number of unique SMILES to generate in enumeration

            `augment_percentage`: float - Percentage of dataset to augment with fusion method
            
            # Returns:
            `List[str]`: Original dataset plus augmented SMILES
            """
            df = pd.read_csv(dataset)
            new_df = augment_dataset(dataset=df, augmentation_methods=augmentation_methods, mask_ratio=mask_ratio, delete_ratio=delete_ratio, 
                                       col_to_augment=col_to_augment, augment_percentage=augment_percentage, seed=seed,
                                       property_col=property_col)
            

            new_df = new_df.drop_duplicates()
            new_df.to_csv("Augmented_QM9.csv", index=True, float_format='%.8e')

            new_data = len(new_df) - len(df)
            print(f"Generated new {new_data} SMILES")

            return new_df
            
    class GraphsModule:
        def __init__(self, parent):
            self.parent = parent

        def augment_data(self, dataset: List[str]):
            pass

    class INCHIModule:
        def __init__(self, parent):
            self.parent = parent    
        
        def augment_data(self, dataset: List[str]):
            pass


if __name__ == '__main__':
    aug = Augmentator(seed=2389)

    augmented_data = aug.SMILES.augment_data(dataset='QM9.csv')
    augmented_data.head()
    
    print(augmented_data)