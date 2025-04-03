from typing import Tuple, List, Optional
from rdkit import Chem
import numpy as np
import re

def atom_positions(smiles: str) -> Tuple[List[str], List[int]]:
    """
    Slice SMILES string into a list of tokens and a list of indices for tokens that do not belong 
    to the special charset.
    
    # Parameters:
    `smiles`: str - SMILES
    """
    charset = set(['[', ']', '(', ')', '=', '#', '%', '.', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '0', '@'])
    
    tokens = list(smiles)
    non_charset_indices = []
    
    for idx, token in enumerate(tokens):
        if token not in charset:
            non_charset_indices.append(idx)

    return tokens, non_charset_indices

def tokenize(smiles: str):
    """
    Slice SMILES string into tokens from a REGEX pattern. This will be used for masking and deleting functions.

    # Parameters:
    `smiles`: str - SMILES
    """

    SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
    regex = re.compile(SMI_REGEX_PATTERN)

    tokens = [token for token in regex.findall(smiles)]
    return tokens


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
    
def enumerateSmiles(smiles: str, num_randomizations: int = 10, max_unique: int = 1000) -> List[str]:
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
        random_smiles = generateRandomSmiles(smiles)

        if random_smiles and random_smiles not in unique_smiles:
            random_mol = Chem.MolFromSmiles(random_smiles)
            if Chem.MolToInchi(original) == Chem.MolToInchi(random_mol):
                unique_smiles.add(random_smiles)
        attempts += 1

    return list(unique_smiles)

def mask(smiles: str, mask_ratio: float = 0.05, attempts: int = 5, seed = 45) -> str:
    """
    Mask a SMILES string with [M] token.
    # Parameters:
    `smiles`: str - SMILES

    `mask_ratio`: float - ratio of tokens to mask
    """
    # É necessário estudar os testes do modelo para caso fazer o mask apenas em strings ou podemos fazer nos índices também.
    # Atualmente está dando o mask em tudo que é possível.
    
    token = '[M]'
    sliced_smiles = tokenize(smiles)

    masked_smiles = set()

    for _ in range(attempts):
        rng = np.random.default_rng(seed.spawn(1)[0])
        masked = sliced_smiles.copy() 
        
        mask_indices = rng.choice(len(masked), int(len(masked) * mask_ratio), replace=False)
        
        for idx in mask_indices:
            masked[idx] = token

        if ''.join(masked) not in masked_smiles:
            masked_smiles.add(''.join(masked))
        else: 
            attempts += 1

    return list(masked_smiles)

def delete(self, smiles: str, delete_ratio: float = 0.3, attempts: int = 5, seed = 45) -> str:
    """
    Delete tokens from a SMILES string.
    # Parameters:
    `smiles`: str - SMILES

    `delete_ratio`: float - ratio of tokens to delete
    """
    deleted_smiles = set()
    sliced_smiles = tokenize(smiles)

    for _ in range(attempts):
        rng = np.random.default_rng(seed.spawn(1)[0])
        deleted = sliced_smiles.copy()
        
        delete_indices = rng.choice(len(deleted), int(len(deleted) * delete_ratio), replace=False)
        
        for idx in delete_indices:
            deleted[idx] = ''

        if ''.join(deleted) not in deleted_smiles:
            deleted_smiles.add(''.join(deleted))
        else:
            attempts += 1

    return list(deleted_smiles)

def swap(smiles: str, attempts: int = 5) -> str:
    """
    Swap two random tokens in a SMILES string.
    # Parameters:
    `smiles`: str - SMILES
    """
    tokens, non_charset_indices = atom_positions(smiles)
    swapped_smiles = set()

    for _ in range(attempts):
        idx1, idx2 = np.random.choice(non_charset_indices, 2, replace=True)
        
        tokens[idx1], tokens[idx2] = tokens[idx2], tokens[idx1]

        if ''.join(tokens) not in swapped_smiles:
            swapped_smiles.add(''.join(tokens))
        else: 
            attempts += 1

    return list(swapped_smiles)

def fusion(smiles: str, mask_ratio: float = 0.05, delete_ratio: float = 0.3) -> str:
    """
    Fusion of mask, delete and swap functions. 0 represents mask, 1 represents delete and 2 represents swap.
    # Parameters:
    `smiles`: str - SMILES

    `mask_ratio`: float - ratio of tokens to mask

    `delete_ratio`: float - ratio of tokens to delete
    """
    chosen = np.random.choice(3, 1)[0]  

    if chosen == 0:
        return mask(smiles, mask_ratio)
    elif chosen == 1:
        return delete(smiles, delete_ratio)
    else:
        return swap(smiles)