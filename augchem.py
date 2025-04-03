import os
from rdkit import Chem
from rdkit import RDLogger
import numpy as np
from typing import List, Tuple, Optional
from modules.smiles.smiles_modules import *

# disable rdkit warnings
RDLogger.logger().setLevel(RDLogger.ERROR)

class Loader:
    def __init__(self, path: str):
        self.path: str = path
        self.data: Optional[List] = None

    def __loadSingleFile(self, file: str) -> List[List[str]]:
        """
        Carrega um arquivo de um dataset
        # Parâmetros:
        `file`: str - caminho do arquivo
        """
        with open(file, 'r') as f:
            # Número de átomos
            natoms: int = int(f.readline())
            f.readline()  # Pula a segunda linha
            
            smiles: List[List[str]] = []

            for num_line, line in enumerate(f):
                if num_line == natoms + 1:
                    smiles.append(line.split())
                    
            return smiles

    def loadDataset(self, directory_path: str, list_mols: List = []) -> List:
        """Load the entire QM9 dataset from a directory containing .xyz files.
        # Parâmetros:
        `directory_path`: str - caminho do diretório contendo os arquivos .xyz

        `list_mols`: list - lista de moléculas
        """

        # Contador para limitar o número de moléculas carregadas para facilitar em testes menores
        cont: int = 0

        for file in os.listdir(directory_path):
            if file.endswith(".xyz") and cont != 150:
                file_path: str = os.path.join(directory_path, file)
                mol: List[List[str]] = self.__loadSingleFile(file_path)
                list_mols.append(mol)

                #print(f"Loaded {file}")
                cont += 1

        return list_mols

    def verifyMolecules(self, list_mols: List) -> Tuple[List[str], List[str]]:
        """Verify if the molecules were correctly imported using RDKit.
        # Parameters:
        `list_mols`: list - list of molecules
        """
        valid_mols: List[str] = []
        invalid_mols: List[str] = []

        for mol in list_mols:
            smiles: str = mol[0][0]
            rdkit_mol = Chem.MolFromSmiles(smiles)
            if rdkit_mol:
                valid_mols.append(smiles)
            else:
                print("Invalid molecule: ", smiles)
                invalid_mols.append(smiles)
                break

        return valid_mols, invalid_mols
    
class Augmentator:
    def __init__(self, seed: int = 4123, dataset: List = []):
        self.dataset = dataset
        self.seed = seed
        self.ss = np.random.SeedSequence(self.seed)
        self.rng = np.random.RandomState(seed=seed)

        self.SMILES = self.SMILESModule(self)

    class SMILESModule:
        def __init__(self, parent):
            self.parent = parent

        def augment_data(self, mask_ratio: float = 0.05, delete_ratio: float = 0.3, num_enumeration_attempts: int = 10, 
                         max_unique: int = 100, augment_percentage: float = 0.2):

            data_to_augment = shuffle_and_split(augment_percentage=augment_percentage, 
                                        dataset=self.parent.dataset)
            augmented_subset = fusion(data_to_augment, mask_ratio=mask_ratio, delete_ratio=delete_ratio)
            augmented_subset.extend(enumerateSmiles(self.parent.dataset, num_randomizations=num_enumeration_attempts, max_unique=max_unique))

            return self.parent.dataset + augmented_subset
            
    class GraphsModule():
        def __init__(self, parent):
            self.parent = parent

    class INCHIModule():
        def __init__(self, parent):
            self.parent = parent    


if __name__ == '__main__':
    aug = Augmentator(seed=23, dataset=['N[C]1C(=C([NH])ON=C1)O'])

    augmented_data = aug.SMILES.augment_data()
    
    print(augmented_data)