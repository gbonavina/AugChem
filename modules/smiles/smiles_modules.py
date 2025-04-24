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

def enumerateSmiles(smiles: str) -> Optional[str]:
    """
    Generate a valid random SMILES string.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Invalid SMILES string: {smiles}")

    random_smiles = Chem.MolToSmiles(mol, canonical=False, doRandom=True)
    random_mol = Chem.MolFromSmiles(random_smiles)
    if random_mol is None:
        return None

    # comparar inhchi a partir de objetos Mol, não de str
    if Chem.MolToInchi(mol) == Chem.MolToInchi(random_mol):
        return random_smiles

    return None

def mask(smiles: str, mask_ratio: float = 0.5, seed = 45) -> List[str]:
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
    
    if isinstance(seed, int):
        rng = np.random.RandomState(seed)
    else:
        rng = seed
    
    sliced_smiles = tokenize(smiles)
    
    masked = sliced_smiles.copy()
    
    mask_indices = rng.choice(len(masked), int(len(masked) * mask_ratio), replace=False)
    
    for idx in mask_indices:
        masked[idx] = token
    
    masked_string = ''.join(masked)
    
    return masked_string

def delete(smiles: str, delete_ratio: float = 0.3, seed = 45) -> List[str]:
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
    
    if isinstance(seed, int):
        rng = np.random.RandomState(seed)
    else:
        rng = seed
    
    sliced_smiles = tokenize(smiles)

    deleted = sliced_smiles.copy()
    
    delete_indices = rng.choice(len(deleted), int(len(deleted) * delete_ratio), replace=False)
    
    for idx in delete_indices:
        deleted[idx] = ''
    
    deleted_string = ''.join(deleted)    
    
    return deleted_string

def swap(smiles: str, seed = 45) -> List[str]:
    """
    Swap two random tokens in SMILES strings.
    
    # Parameters:
    `smiles`: str - SMILES string to augment

    `attempts`: int - number of swapping attempts per SMILES

    `seed`: int or numpy.random.Generator - random seed for reproducibility
    
    # Returns:
    `List[str]`: List of SMILES strings with tokens swapped
    """

    if isinstance(seed, int):
        rng = np.random.RandomState(seed)
    else:
        rng = seed
    
    tokens, non_charset_indices = atom_positions(smiles)
    swapped = tokens.copy()

    idx1, idx2 = rng.choice(non_charset_indices, 2, replace=False)
    swapped[idx1], swapped[idx2] = swapped[idx2], swapped[idx1]

    swapped_string = ''.join(swapped)
       
    return swapped_string

def fusion(smiles: str, mask_ratio: float = 0.05, delete_ratio: float = 0.3, seed = 45) -> List[str]:
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

    if hasattr(seed, 'choice') and callable(seed.choice):
        rng = seed
    else:
        rng = np.random.RandomState(seed)
    
    if not smiles:
        raise ValueError("Empty SMILES string isn't valid.")
    
    chosen = rng.choice(3, 1)[0]
    
    try:
        if chosen == 0:
            augmented = mask(smiles, mask_ratio=mask_ratio, seed=rng)
        elif chosen == 1:
            augmented = delete(smiles, delete_ratio=delete_ratio, seed=rng)
        else:
            augmented = swap(smiles, seed=rng)
    
    except Exception as e:
        print(f"Error during augmentation of {smiles}: {str(e)}")
        raise ValueError(e) 
    
    return augmented

# mandar um email pro quiles pedindo um modelo lstm para testar o código
# fazer a seleçao na hora de aplicar o método 
def augment_dataset(col_to_augment: str, dataset: pd.DataFrame, augmentation_methods: List[str], mask_ratio: float = 0.1, property_col: str = None, delete_ratio: float = 0.3,
                     augment_percentage: float = 0.2, seed: int = 42):

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
        try:
            augmented_smiles: List[str] = []
            for method in augmentation_methods:
                if method == "mask":
                    row_to_augment = rng.randint(low=0, high=(len(dataset)-1))
                    original_idx = working_copy.index[row_to_augment]
                    row = working_copy.iloc[row_to_augment].copy()
                    
                    smiles = row[col_to_augment]
                    # print(f"Augmenting {smiles} with {method} method.")

                    augmented_smiles.append(mask(
                        smiles,
                        mask_ratio=mask_ratio,
                        seed=rng
                    ))
                elif method == "delete":
                    row_to_augment = rng.randint(low=0, high=(len(dataset)-1))
                    original_idx = working_copy.index[row_to_augment]
                    row = working_copy.iloc[row_to_augment].copy()
                    
                    smiles = row[col_to_augment]
                    # print(f"Augmenting {smiles} with {method} method.")

                    augmented_smiles.append(delete(
                        smiles,
                        delete_ratio=delete_ratio,
                        seed=rng
                    ))
                elif method == "swap":
                    row_to_augment = rng.randint(low=0, high=(len(dataset)-1))
                    original_idx = working_copy.index[row_to_augment]
                    row = working_copy.iloc[row_to_augment].copy()
                    
                    smiles = row[col_to_augment]
                    # print(f"Augmenting {smiles} with {method} method.")

                    augmented_smiles.append(swap(
                        smiles,
                        seed=rng
                    ))
                elif method == "fusion":
                    row_to_augment = rng.randint(low=0, high=(len(dataset)-1))
                    original_idx = working_copy.index[row_to_augment]
                    row = working_copy.iloc[row_to_augment].copy()
                    
                    smiles = row[col_to_augment]
                    # print(f"Augmenting {smiles} with {method} method.")

                    augmented_smiles.append(fusion(
                        smiles,
                        mask_ratio=mask_ratio,
                        delete_ratio=delete_ratio,
                        seed=rng
                    ))
                elif method == "enumeration":
                    row_to_augment = rng.randint(low=0, high=(len(dataset)-1))
                    original_idx = working_copy.index[row_to_augment]
                    row = working_copy.iloc[row_to_augment].copy()
                    
                    smiles = row[col_to_augment]
                    # print(f"Augmenting {smiles} with {method} method.")
                    
                    augmented_smiles.append(enumerateSmiles(
                        smiles
                    ))
                else:
                    raise ValueError(f"Unknown augmentation method: {method}")

            augmented_smiles = list(dict.fromkeys(augmented_smiles))
            augmented_smiles = augmented_smiles[: target_new_rows - augmented_count]

            for aug_smiles in augmented_smiles:
                new_row = row.copy()
                new_row[col_to_augment] = aug_smiles

                for prop_col in [c for c in new_row.index if c.startswith("Property_")]:
                    new_row[prop_col] = "-"
                new_row["parent_idx"] = original_idx
                new_rows.append(new_row)
                augmented_count += 1
                if augmented_count >= target_new_rows:
                    break

            if augmented_count >= target_new_rows:
                break

        except Exception:
            continue
         
    filtered_df = dataset[[col_to_augment, property_col]].copy()

    if new_rows:
        new_data = pd.DataFrame(new_rows)
        augmented_df = pd.concat([filtered_df, new_data], ignore_index=True)
        augmented_df = augmented_df.fillna("-1")
        
    return augmented_df
    
