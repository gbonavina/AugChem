import os
from rdkit import Chem
#import deepchem
import numpy as np
from typing import List, Tuple, Optional
import re

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
    def __init__(self, seed: int = 4123):
        self.seed = seed
        self.ss = np.random.SeedSequence(self.seed)

    def tokenize(self, smiles: str):
        """
        Slice SMILES string into tokens from a REGEX pattern. This will be used for masking and deleting functions.

        # Parameters:
        `smiles`: str - SMILES
        """

        SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
        regex = re.compile(SMI_REGEX_PATTERN)

        tokens = [token for token in regex.findall(smiles)]
        return tokens
    
    def atom_positions(self, smiles: str) -> Tuple[List[str], List[int]]:
        """
        Slice SMILES string into a list of tokens and a list of indices for tokens that do not belong 
        to the special charset.
        
        # Parameters:
        `smiles`: str - SMILES
        """
        # Conjunto de caracteres especiais a serem tratados separadamente
        charset = set(['[', ']', '(', ')', '=', '#', '%', '.', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '0', '@'])
        
        tokens = list(smiles)
        non_charset_indices = []
        
        # Usamos enumerate para obter tanto o índice quanto o token
        for idx, token in enumerate(tokens):
            if token not in charset:
                non_charset_indices.append(idx)

        return tokens, non_charset_indices
    
    def mask(self, smiles: str, mask_ratio: float = 0.05, attempts: int = 5) -> str:
        """
        Mask a SMILES string with [M] token.
        # Parameters:
        `smiles`: str - SMILES

        `mask_ratio`: float - ratio of tokens to mask
        """
        # É necessário estudar os testes do modelo para caso fazer o mask apenas em strings ou podemos fazer nos índices também.
        # Atualmente está dando o mask em tudo que é possível.
        
        token = '[M]'
        sliced_smiles = self.tokenize(smiles)

        masked_smiles = set()

        for _ in range(attempts):
            # Criar um novo gerador de números aleatórios para garantir variação
            rng = np.random.default_rng(self.ss.spawn(1)[0])
            masked = sliced_smiles.copy() 
            
            # Gerar os índices de forma aleatória com base no novo RNG
            mask_indices = rng.choice(len(masked), int(len(masked) * mask_ratio), replace=False)
            
            for idx in mask_indices:
                masked[idx] = token

            if ''.join(masked) not in masked_smiles:
                masked_smiles.add(''.join(masked))
            else: 
                # Se o SMILES mascarado já estiver no conjunto, tente novamente
                attempts += 1

        return list(masked_smiles)   
    
    def delete(self, smiles: str, delete_ratio: float = 0.3, attempts: int = 5) -> str:
        """
        Delete tokens from a SMILES string.
        # Parameters:
        `smiles`: str - SMILES

        `delete_ratio`: float - ratio of tokens to delete
        """
        deleted_smiles = set()
        sliced_smiles = self.tokenize(smiles)

        for _ in range(attempts):
            # Criar um novo gerador de números aleatórios para garantir variação
            rng = np.random.default_rng(self.ss.spawn(1)[0])
            deleted = sliced_smiles.copy()
            
            # Gerar os índices de forma aleatória com base no novo RNG
            delete_indices = rng.choice(len(deleted), int(len(deleted) * delete_ratio), replace=False)
            
            for idx in delete_indices:
                deleted[idx] = ''

            if ''.join(deleted) not in deleted_smiles:
                deleted_smiles.add(''.join(deleted))
            else:
                # Se o SMILES excluído já estiver no conjunto, tente novamente
                attempts += 1

        return list(deleted_smiles)
    
    def swap(self, smiles: str, attempts: int = 5) -> str:
        """
        Swap two random tokens in a SMILES string.
        # Parameters:
        `smiles`: str - SMILES
        """
        # Tokenize the SMILES string
        # In this case we're not swapping special characters, but it's a thing we should study if it affects the model or not.
        tokens, non_charset_indices = self.atom_positions(smiles)
        swapped_smiles = set()

        for _ in range(attempts):
            # Randomly select two indices to swap
            idx1, idx2 = np.random.choice(non_charset_indices, 2, replace=True)
            
            # Swap the tokens
            tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]

            if ''.join(tokens) not in swapped_smiles:
                swapped_smiles.add(''.join(tokens))
            else: 
                # If the swapped SMILES is already in the set, try again
                attempts += 1

        return list(swapped_smiles)
    
    def fusion(self, smiles: str, mask_ratio: float = 0.05, delete_ratio: float = 0.3) -> str:
        """
        Fusion of mask, delete and swap functions. 0 represents mask, 1 represents delete and 2 represents swap.
        # Parameters:
        `smiles`: str - SMILES

        `mask_ratio`: float - ratio of tokens to mask

        `delete_ratio`: float - ratio of tokens to delete
        """
        chosen = np.random.choice(3, 1)[0]  # Ensure chosen is an integer

        if chosen == 0:
            return self.mask(smiles, mask_ratio)
        elif chosen == 1:
            return self.delete(smiles, delete_ratio)
        else:
            return self.swap(smiles)
        
    @staticmethod
    def generateRandomSmiles(smiles: str, attempts: int = 100) -> Optional[str]:
        """
        Generate a valid random SMILES string.
        # Params:
        `smiles`: str - SMILES

        `attempts`: int - number of attempts to generate a random SMILES string
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        for _ in range(attempts):
            # Generate a random SMILES string
            random_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)

            if Chem.MolFromSmiles(random_smiles):
                return random_smiles
            
        return None
    
    def enumerateSmiles(self, smiles: str, num_randomizations: int = 10, max_unique: int = 1000) -> List[str]:
        """
        Create multiple representations of a SMILES string.
        # Parameters:
        `smiles`: str - SMILES

        `num_randomizations`: int - number of attempts to generate random SMILES strings

        `max_unique`: int - maximum number of unique SMILES strings to generate
        """
        unique_smiles = set()
        original = Chem.MolFromSmiles(smiles)
        if original is None:
            raise ValueError(f"Invalid SMILES string: {smiles}")
        
        attempts = 0
        while len(unique_smiles) < max_unique and attempts < num_randomizations:
            random_smiles = self.generateRandomSmiles(smiles)

            if random_smiles and random_smiles not in unique_smiles:
                random_mol = Chem.MolFromSmiles(random_smiles)
                if Chem.MolToInchi(original) == Chem.MolToInchi(random_mol):
                    unique_smiles.add(random_smiles)
            attempts += 1

        return list(unique_smiles)
    
if __name__ == '__main__':
    aug = Augmentator(seed=23)
    smiles = aug.mask('N[C]1C(=C([NH])ON=C1)O', mask_ratio=0.15)
    print(smiles)