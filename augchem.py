import os
from rdkit import Chem
import numpy as np
from typing import List, Tuple, Optional

class Loader:
    def __init__(self, path: str):
        self.path: str = path
        self.data: Optional[List] = None

    def loadSingleFile(self, file: str) -> List[List[str]]:
        """
        Carrega um arquivo de um dataset
        # Parâmetros:
        file: str - caminho do arquivo
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
        directory_path: str - caminho do diretório contendo os arquivos .xyz
        list_mols: list - lista de moléculas
        """

        # Contador para limitar o número de moléculas carregadas para facilitar em testes menores
        cont: int = 0

        for file in os.listdir(directory_path):
            if file.endswith(".xyz") and cont != 50:
                file_path: str = os.path.join(directory_path, file)
                mol: List[List[str]] = self.loadSingleFile(file_path)
                list_mols.append(mol)

                #print(f"Loaded {file}")
                cont += 1

        return list_mols

    def verifyMolecules(self, list_mols: List) -> Tuple[List[str], List[str]]:
        """Verifica se as moléculas foram importadas corretamente usando RDKit.
        # Parâmetros:
        list_mols: list - lista de moléculas
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
    def __init__(self, seed: int):
        self.seed: int = seed
        np.random.seed(self.seed)

    def slice_smiles(self, smiles: str) -> List[str]:
        """
        Slice a SMILES string into tokens.
        # Params:
        smiles: str - SMILES
        """
        sliced_smiles: List[str] = []
        i: int = 0
        while i < len(smiles):
            if smiles[i] == '(' or smiles[i] == '[':
                # Handle parentheses and brackets
                end: int = smiles.find(')', i) if smiles[i] == '(' else smiles.find(']', i)
                sliced_smiles.append(smiles[i:end+1])
                i = end + 1
            elif smiles[i].isalpha():
                # Handle atoms (single letter or two letters)
                if i + 1 < len(smiles) and smiles[i+1].islower():
                    atom: str = smiles[i:i+2]
                    i += 2
                else:
                    atom: str = smiles[i]
                    i += 1
                
                # Check for numbers after the atom
                while i < len(smiles) and smiles[i].isdigit():
                    atom += smiles[i]
                    i += 1
                
                sliced_smiles.append(atom)
            else:
                # Handle other characters (=, #, etc.)
                sliced_smiles.append(smiles[i])
                i += 1

        return sliced_smiles
    
    def tokenizer(self, smiles: str) -> Tuple[List[str], List[int]]:
        # Tokenize the SMILES string
        charset: set = set('()[]=#')
        tokens: List[str] = []
        non_charset_indices: List[int] = []

        i: int = 0
        while i < len(smiles):
            if smiles[i] in charset:
                tokens.append(smiles[i])
                i += 1
            elif smiles[i].isalpha():
                if i + 1 < len(smiles) and smiles[i+1].islower():
                    tokens.append(smiles[i:i+2])
                    i += 2
                else:
                    tokens.append(smiles[i])
                    i += 1
                non_charset_indices.append(len(tokens) - 1)
            elif smiles[i].isdigit():
                num: str = ''
                while i < len(smiles) and smiles[i].isdigit():
                    num += smiles[i]
                    i += 1
                tokens.append(num)
                non_charset_indices.append(len(tokens) - 1)
            else:
                tokens.append(smiles[i])
                i += 1

        return tokens, non_charset_indices
    
    def mask(self, smiles: str, mask_ratio: float = 0.05) -> str:
        """
        Mask a SMILES string with [M] token.
        # Params:
        smiles: str - SMILES

        mask_ratio: float - ratio of tokens to mask
        """
        token: str = '[M]'
        sliced_smiles: List[str] = self.slice_smiles(smiles)        

        mask_indices: np.ndarray = np.random.choice(len(sliced_smiles), int(len(sliced_smiles) * mask_ratio), replace=False)
        
        for idx in mask_indices:
            sliced_smiles[idx] = token

        return ''.join(sliced_smiles)   
    
    def delete(self, smiles: str, delete_ratio: float = 0.3) -> str:
        """
        Delete tokens from a SMILES string.
        # Params:
        smiles: str - SMILES

        delete_ratio: float - ratio of tokens to delete
        """
        sliced_smiles: List[str] = self.slice_smiles(smiles)

        delete_indices: np.ndarray = np.random.choice(len(sliced_smiles), int(len(sliced_smiles) * delete_ratio), replace=False)
        
        for idx in delete_indices:
            sliced_smiles[idx] = ''

        return ''.join(sliced_smiles)
    
    def swap(self, smiles: str) -> str:
        """
        Swap two random tokens in a SMILES string.
        # Params:
        smiles: str - SMILES
        """
        # Tokenize the SMILES string
        tokens, non_charset_indices = self.tokenizer(smiles)
        
        # Randomly select two indices to swap
        idx1, idx2 = np.random.choice(non_charset_indices, 2, replace=True)
        
        # Swap the tokens
        tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]

        return ''.join(tokens)
    
    def fusion(self, smiles: str, mask_ratio: float = 0.05, delete_ratio: float = 0.3) -> str:
        """
        Fusion of mask, delete and swap functions. 0 represents mask, 1 represents delete and 2 represents swap.
        # Params:
        smiles: str - SMILES

        mask_ratio: float - ratio of tokens to mask

        delete_ratio: float - ratio of tokens to delete
        """
        chosen: int = np.random.choice(3, 1)[0]  # Ensure chosen is an integer

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
        smiles: str - SMILES
        attempts: int - number of attempts to generate a random SMILES string
        """
        mol = Chem.MolFromSmiles(smiles)
        
        for _ in range(attempts):
            # Generate a random SMILES string
            random_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)

            if Chem.MolFromSmiles(random_smiles):
                return random_smiles
            
        return None
    
    def enumerateSmiles(self, smiles: str, num_randomizations: int = 10, max_unique: int = 1000) -> List[str]:
        """
        Create multiple representations of a SMILES string.
        # Params:
        smiles: str - SMILES
        num_randomizations: int - number of attempts to generate random SMILES strings
        max_unique: int - maximum number of unique SMILES strings to generate
        """
        unique_smiles = set()
        original = Chem.MolFromSmiles(smiles)
        
        attempts = 0
        while len(unique_smiles) < max_unique and attempts < num_randomizations:
            random_smiles = self.generateRandomSmiles(smiles)

            if random_smiles and random_smiles not in unique_smiles:
                random_mol = Chem.MolFromSmiles(random_smiles)
                if Chem.MolToInchi(original) == Chem.MolToInchi(random_mol):
                    unique_smiles.add(random_smiles)
            attempts += 1

        return list(unique_smiles)
