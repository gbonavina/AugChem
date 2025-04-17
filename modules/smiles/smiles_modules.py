from typing import Tuple, List, Optional
from rdkit import Chem
from rdkit import RDLogger
import numpy as np
import pandas as pd
import ast
import re

# disable rdkit warnings
RDLogger.DisableLog('rdApp.*')


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
    Create multiple representations of a SMILES string or list of SMILES strings.
    
    # Parameters:
    `smiles`: str or List[str] - SMILES string(s)
    `num_randomizations`: int - number of attempts to generate random SMILES strings
    `max_unique`: int - maximum number of unique SMILES strings to generate
    
    # Returns:
    `List[str]`: List of enumerated SMILES strings
    """
    all_unique_smiles = []
    
    unique_smiles = set()
    original = Chem.MolFromSmiles(smiles)  
    
    attempts = 0
    while len(unique_smiles) < max_unique and attempts < num_randomizations:
        random_smiles = generateRandomSmiles(smiles)
        
        if random_smiles and random_smiles not in unique_smiles:
            random_mol = Chem.MolFromSmiles(random_smiles)
            if Chem.MolToInchi(original) == Chem.MolToInchi(random_mol):
                unique_smiles.add(random_smiles)
        attempts += 1
    
    all_unique_smiles.extend(list(unique_smiles))

    return all_unique_smiles

def mask(smiles: str, mask_ratio: float = 0.5, attempts: int = 15, seed = 45) -> List[str]:
    """
    Mask tokens in a SMILES string with [M] token.
    
    # Parameters:
    `smiles`: str - SMILES string to augment
    `mask_ratio`: float - ratio of tokens to mask
    `attempts`: int - number of masking attempts
    `seed`: int or numpy.random.Generator - random seed for reproducibility
    
    # Returns:
    `List[str]`: List of masked SMILES strings
    """
    token = '[M]'
    all_masked_smiles = []
    
    if isinstance(seed, int):
        rng = np.random.RandomState(seed)
    else:
        rng = seed
    
    sliced_smiles = tokenize(smiles)
    masked_smiles = set()
    
    remaining_attempts = attempts
    while len(masked_smiles) < attempts and remaining_attempts > 0:
        masked = sliced_smiles.copy()
        
        mask_indices = rng.choice(len(masked), int(len(masked) * mask_ratio), replace=False)
        
        for idx in mask_indices:
            masked[idx] = token
        
        masked_string = ''.join(masked)
        if masked_string not in masked_smiles:
            masked_smiles.add(masked_string)
        
        remaining_attempts -= 1
    
    all_masked_smiles.extend(list(masked_smiles))
    
    return all_masked_smiles

def delete(smiles: str, delete_ratio: float = 0.3, attempts: int = 5, seed = 45) -> List[str]:
    """
    Delete tokens from SMILES strings.
    
    # Parameters:
    `smiles`: str - SMILES string to augment
    `delete_ratio`: float - ratio of tokens to delete
    `attempts`: int - number of deletion attempts per SMILES
    `seed`: int or numpy.random.Generator - random seed for reproducibility
    
    # Returns:
    `List[str]`: List of SMILES strings with tokens deleted
    """
    all_deleted_smiles = []
    
    if isinstance(seed, int):
        rng = np.random.RandomState(seed)
    else:
        rng = seed
    
    sliced_smiles = tokenize(smiles)
    deleted_smiles = set()
    
    remaining_attempts = attempts
    while len(deleted_smiles) < attempts and remaining_attempts > 0:
        deleted = sliced_smiles.copy()
        
        delete_indices = rng.choice(len(deleted), int(len(deleted) * delete_ratio), replace=False)
        
        for idx in delete_indices:
            deleted[idx] = ''
        
        deleted_string = ''.join(deleted)
        if deleted_string not in deleted_smiles:
            deleted_smiles.add(deleted_string)
        
        remaining_attempts -= 1
    
    all_deleted_smiles.extend(list(deleted_smiles))
    
    return all_deleted_smiles

def swap(smiles: str, attempts: int = 5, seed = 45) -> List[str]:
    """
    Swap two random tokens in SMILES strings.
    
    # Parameters:
    `smiles`: str - SMILES string to augment

    `attempts`: int - number of swapping attempts per SMILES

    `seed`: int or numpy.random.Generator - random seed for reproducibility
    
    # Returns:
    `List[str]`: List of SMILES strings with tokens swapped
    """
    all_swapped_smiles = []

    if isinstance(seed, int):
        rng = np.random.RandomState(seed)
    else:
        rng = seed
    
    tokens, non_charset_indices = atom_positions(smiles)
    swapped_smiles = set()
    
    if len(non_charset_indices) < 2:
        all_swapped_smiles.append(smiles)
    
    remaining_attempts = attempts
    while len(swapped_smiles) < attempts and remaining_attempts > 0:
        swapped = tokens.copy()
        
        idx1, idx2 = rng.choice(non_charset_indices, 2, replace=False)
        
        swapped[idx1], swapped[idx2] = swapped[idx2], swapped[idx1]
        
        swapped_string = ''.join(swapped)
        if swapped_string not in swapped_smiles:
            swapped_smiles.add(swapped_string)
        
        remaining_attempts -= 1
    
    all_swapped_smiles.extend(list(swapped_smiles))
    
    return all_swapped_smiles

def fusion(smiles: str, mask_ratio: float = 0.05, delete_ratio: float = 0.3, attempts: int = 5, seed = 45) -> List[str]:
    """
    Fusion of mask, delete and swap functions. Randomly choose one of the three augmentation methods.
    
    # Parameters:
    `smiles`: str - SMILES string to augment
    `mask_ratio`: float - ratio of tokens to mask
    `delete_ratio`: float - ratio of tokens to delete
    `attempts`: int - number of augmentation attempts
    `seed`: int or numpy.random.RandomState - random seed for reproducibility
    
    # Returns:
    `List[str]`: List of augmented SMILES strings
    """
    augmented_smiles = []
    
    if hasattr(seed, 'choice') and callable(seed.choice):
        rng = seed
    else:
        rng = np.random.RandomState(seed)
    
    if not smiles:
        raise ValueError("Empty SMILES string isn't valid.")
    
    chosen = rng.choice(3, 1)[0]
    
    try:
        if chosen == 0:
            augmented = mask(smiles, mask_ratio=mask_ratio, attempts=attempts, seed=rng)
        elif chosen == 1:
            augmented = delete(smiles, delete_ratio=delete_ratio, attempts=attempts, seed=rng)
        else:
            augmented = swap(smiles, attempts=attempts, seed=rng)
        
        augmented_smiles.extend(augmented)
    
    except Exception as e:
        print(f"Error during augmentation of {smiles}: {str(e)}")
        raise ValueError(e) 
    
    return augmented_smiles


# seleciona a coluna ai retorna a coluna com os dados aumentados e a propriedade como eu havia feito antes.
def augment_dataset(col_to_augment: str, dataset: pd.DataFrame, augmentation_methods: List[str], mask_ratio: float = 0.1, property_col: str = None, delete_ratio: float = 0.3, attempts: int = 10,
                     augment_percentage: float = 0.2, seed: int = 42, max_unique: int = 100):

    # if col_to_augment.startswith("INCHI_") or dataset[col_to_augment][0].startswith("InChI="):
    #     raise ValueError("Input appears to be in InChI format. This function only works with SMILES format.")

    try:
        mol = Chem.MolFromSmiles(dataset[col_to_augment][0])
    except Exception as e:
        raise ValueError("Input appears to be in the wrong format. This function only works with SMILES format.")

    rng = np.random.RandomState(seed)

    if property_col:
        working_copy = dataset[[col_to_augment, property_col]].copy()
    else:
        working_copy = dataset[[col_to_augment]].copy()

    target_new_rows = int(len(dataset) * augment_percentage)

    new_rows = []
    augmented_count = 0

    while augmented_count < target_new_rows:
        row_to_augment = rng.randint(low=0, high=(len(dataset)-1))
        original_idx = working_copy.index[row_to_augment]
        row = working_copy.iloc[row_to_augment].copy()
        
        smiles = row[col_to_augment]

        try: 
            if "mask" in augmentation_methods:
                augmented_smiles = mask(smiles, mask_ratio=mask_ratio, attempts=attempts, seed=rng)
            if "delete" in augmentation_methods:
                augmented_smiles = delete(smiles, delete_ratio=delete_ratio, attempts=attempts, seed=rng)
            if "swap" in augmentation_methods:
                augmented_smiles = swap(smiles, attempts=attempts, seed=rng)
            if "fusion" in augmentation_methods:
                augmented_smiles = fusion(smiles, mask_ratio=mask_ratio, delete_ratio=delete_ratio, 
                                        attempts=attempts, seed=rng)
            if "enumeration" in augmentation_methods:
                augmented_smiles = enumerateSmiles(smiles, num_randomizations=attempts)
            else:
                raise ValueError(f"Unknown augmentation methods: {augmentation_methods}")
            
            if augmented_smiles:
                for aug_smiles in augmented_smiles:
                    new_row = row.copy()
                    new_row[col_to_augment] = aug_smiles
                    
                    property_columns = [col for col in new_row.index if col.startswith('Property_')]
                    for prop_col in property_columns:
                        new_row[prop_col] = "-"
                
                    new_row['parent_idx'] = original_idx
                    
                    new_rows.append(new_row)
                    augmented_count += 1
            
                if augmented_count >= target_new_rows:
                    break

        except Exception as e:
            # print(f"Error augmenting SMILES {smiles}: {str(e)}")
            continue
         
    filtered_df = dataset[[col_to_augment, property_col]].copy()

    if new_rows:
        new_data = pd.DataFrame(new_rows)
        augmented_df = pd.concat([filtered_df, new_data], ignore_index=True)
        
    return augmented_df
    
