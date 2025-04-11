import os
from rdkit import Chem
from rdkit import RDLogger
import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Required
from pathlib import Path
from modules.smiles.smiles_modules import *

# disable rdkit warnings
RDLogger.logger().setLevel(RDLogger.ERROR)

class Loader:
    def __init__(self, path):
        self.path = path

    def load_qm9_xyz(self):
        """Load a single QM9.xyz file."""
        with open(self.path, 'r') as f:
            # Number of atoms
            natoms = int(f.readline())
            # Properties are in the second line
            properties = list(map(float, f.readline().split()[2:]))
            # Read atomic coordinates and types
            atoms = []
            coordinates = []
            smiles = []
            # print(properties)
            for num_line, line in enumerate(f):
                # print(num_line, line)
                if num_line >= 0 and num_line < natoms:
                    info = line.replace("*^","e").split()
                    atoms.append(info[0])
                    coordinates.append(list(map(float, info[1:-1])))
                if num_line == natoms + 1:
                    smiles.append(line.split())
        return {
            "natoms": natoms,
            "atoms": atoms,
            "coordinates": np.array(coordinates),
            "smiles": smiles,
            "properties": properties
        }

    def load_qm9_dataset(self, list_mols=[]):
        """Load the entire QM9 dataset from a directory containing .xyz files."""
        X = []
        Y = []
        S = []
        SMILES = []
        for file_name in os.listdir(self.path):
            if file_name.endswith(".xyz"):
                file_path = os.path.join(self.path, file_name)
                molecule_data = self.load_qm9_xyz(file_path)
                if molecule_data['natoms'] in list_mols or len(list_mols)==0:
                    X.append([molecule_data['atoms'], molecule_data['coordinates']])
                    Y.append(molecule_data['properties'])
                    S.append(molecule_data['natoms'])
                    SMILES.append(molecule_data['smiles'])

        return Y, SMILES
    
    def qm9_to_csv(self):
        Y, SMILES = self.load_qm9_dataset()

        Y = Y.tolist()

        # Lista para armazenar todos os pares SMILES/propriedade
        todos_smiles = []
        todas_propriedades = []

        # Para cada molécula
        for i, propriedades in enumerate(Y):
            # Obter a propriedade 0 desta molécula
            propriedade_0 = propriedades[0]
            
            # Obter os SMILES desta molécula
            smiles_lista = SMILES[i]
            
            # Adicionar cada SMILES com a propriedade 0 correspondente
            for smiles in smiles_lista:
                todos_smiles.append(smiles)
                todas_propriedades.append(propriedade_0)

        # Criar dataframe com todos os pares
        dataframe = pd.DataFrame(
            {
                'SMILES': todos_smiles,
                'Property_0': todas_propriedades
            }
        )

        dataframe.to_csv('QM9.csv', index=False, float_format='%.8e')

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
        def augment_data(self, dataset: Path, mask_ratio: float = 0.1, delete_ratio: float = 0.3, num_enumeration_attempts: int = 10, seed: int = 42, 
                            max_unique: int = 100, augment_percentage: float = 0.2, augmentation_methods: List[str] = ["fusion", "enumerate"], smiles_col: str = 'SMILES') -> pd.DataFrame:
            """
            Augment SMILES strings using fusion and enumeration methods.
            
            # Parameters:
            `dataset`: Path - List of SMILES strings to augment

            `mask_ratio`: float - Ratio of tokens to mask in fusion method

            `delete_ratio`: float - Ratio of tokens to delete in fusion method

            `num_enumeration_attempts`: int - Number of attempts for SMILES enumeration

            `max_unique`: int - Maximum number of unique SMILES to generate in enumeration

            `augment_percentage`: float - Percentage of dataset to augment with fusion method
            
            # Returns:
            `List[str]`: Original dataset plus augmented SMILES
            """
            df = pd.read_csv(dataset)
            new_df = augment_dataset(dataset=df, augmentation_methods=augmentation_methods, mask_ratio=mask_ratio, delete_ratio=delete_ratio, 
                                       attempts=num_enumeration_attempts, smiles_collumn=smiles_col, augment_percentage=augment_percentage, seed=seed)
            

            new_df.to_csv("Augmented_QM9.csv", index=False, float_format='%.8e')

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