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

class Augmentator:        
    """
    Main class for molecular data augmentation across multiple representation formats.
    
    This class provides a unified interface for augmenting molecular data in different 
    formats through specialized modules. It maintains consistent random state management
    across all modules for reproducible augmentation results.
    
    The class contains three specialized modules:
    - SMILESModule: For augmenting SMILES string representations of molecules
    - GraphsModule: For augmenting molecular graph representations
    - INCHIModule: For augmenting InChI string representations of molecules
    
    Parameters
    ----------
    `seed` : int, default=4123
        Random seed for initializing the random number generators used across
        all augmentation modules
        
    Attributes
    ----------
    `seed` : int
        The random seed used for initialization
    `ss` : numpy.random.SeedSequence
        Seed sequence for generating independent random streams
    `rng` : numpy.random.RandomState
        Random number generator for reproducible random operations
    `SMILES` : SMILESModule
        Module containing methods for SMILES-based augmentation
    `Graphs` : GraphsModule
        Module containing methods for molecular graph-based augmentation
    `INCHI` : INCHIModule
        Module containing methods for InChI-based augmentation
    
    Examples
    --------
    >>> augmentator = Augmentator(seed=42)
    >>> augmented_data = augmentator.SMILES.augment_data(dataset='molecules.csv')
    """
    def __init__(self, seed: int = 42):
        """
        Initialize an Augmentator instance with the specified random seed.
        
        Sets up random number generation infrastructure and initializes the
        specialized augmentation modules.
        
        Parameters
        ----------
        `seed` : int, default=4123
            Random seed for initializing random number generators
        """
        self.seed = seed
        self.ss = np.random.SeedSequence(self.seed)
        self.rng = np.random.RandomState(seed=seed)

        self.SMILES = self.SMILESModule(self)
        self.Graphs = self.GraphsModule(self)
        self.INCHI = self.INCHIModule(self)

    class SMILESModule:
        """
        Module for augmenting molecular data in SMILES format.
        
        Provides methods for generating augmented SMILES representations using various
        techniques including masking, deletion, swapping, fusion, and enumeration.
        """
        def __init__(self, parent):
            self.parent = parent

        # Receber csv, colocar coluna extra de qual id veio o aumento, adicionar tag de qual metodo vai ser utilizado e aumento de %: inicial 1000 dados -> 1200 dados, essa é a %
        def augment_data(self, dataset: Path, mask_ratio: float = 0.1, delete_ratio: float = 0.3, seed: int = 42, 
                            augment_percentage: float = 0.2, augmentation_methods: List[str] = ["fusion", "enumerate"], col_to_augment: str = 'SMILES',
                            property_col: str = None) -> pd.DataFrame:
            """
            Augment molecular SMILES data from a CSV file.
            
            Reads SMILES strings from a CSV file, applies specified augmentation methods,
            and returns the augmented dataset. Also saves the augmented dataset to a new CSV file.
            
            Parameters
            ----------
            `dataset` : Path
                Path to the CSV file containing SMILES data to augment

            `mask_ratio` : float, default=0.1
                Fraction of tokens to mask when using masking-based augmentation methods

            `delete_ratio` : float, default=0.3
                Fraction of tokens to delete when using deletion-based augmentation methods

            `seed` : int, default=42
                Random seed for reproducible augmentation

            `augment_percentage` : float, default=0.2
                Target size of augmented dataset as a fraction of original dataset size

            `augmentation_methods` : List[str], default=["fusion", "enumerate"]
                List of augmentation methods to apply. Valid options include: 
                "mask", "delete", "swap", "fusion", "enumeration"

            `col_to_augment` : str, default='SMILES'
                Column name in the CSV file containing SMILES strings to augment

            `property_col` : str, optional
                Column name containing property values to preserve in augmented data
                
            Returns
            -------
            `pd.DataFrame`
                DataFrame containing both original and augmented molecules, with a 'parent_idx'
                column linking augmented molecules to their source molecules
                
            Notes
            -----
            The augmented dataset is automatically saved to "Augmented_QM9.csv" in the
            current working directory.
            """
            df = pd.read_csv(dataset)
            new_df = augment_dataset(dataset=df, augmentation_methods=augmentation_methods, mask_ratio=mask_ratio, delete_ratio=delete_ratio, 
                                       col_to_augment=col_to_augment, augment_percentage=augment_percentage, seed=seed,
                                       property_col=property_col)
            

            new_df = new_df.drop_duplicates()
            new_df.to_csv(f"Augmented_{dataset}", index=True, float_format='%.8e')

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