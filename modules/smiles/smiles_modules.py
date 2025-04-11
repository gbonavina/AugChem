from typing import Tuple, List, Optional
from rdkit import Chem
import numpy as np
import re
import pandas as pd
import ast

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
    
def enumerateSmiles(smiles, num_randomizations: int = 10, max_unique: int = 1000) -> List[str]:
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
    
    if isinstance(smiles, str):
        smiles_list = [smiles]
    else:
        smiles_list = smiles
    
    for single_smiles in smiles_list:
        unique_smiles = set()
        original = Chem.MolFromSmiles(single_smiles)
        if original is None:
            continue  
        
        attempts = 0
        while len(unique_smiles) < max_unique and attempts < num_randomizations:
            random_smiles = generateRandomSmiles(single_smiles)
            
            if random_smiles and random_smiles not in unique_smiles:
                random_mol = Chem.MolFromSmiles(random_smiles)
                if Chem.MolToInchi(original) == Chem.MolToInchi(random_mol):
                    unique_smiles.add(random_smiles)
            attempts += 1
        
        all_unique_smiles.extend(list(unique_smiles))
    
    return all_unique_smiles

def mask(dataset: List, mask_ratio: float = 0.5, attempts: int = 15, seed = 45) -> List[str]:
    """
    Mask tokens in SMILES strings with [M] token.
    
    # Parameters:
    `dataset`: List - List of SMILES strings
    `mask_ratio`: float - ratio of tokens to mask
    `attempts`: int - number of masking attempts per SMILES
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
    
    for smiles in dataset:
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

def delete(dataset: List, delete_ratio: float = 0.3, attempts: int = 5, seed = 45) -> List[str]:
    """
    Delete tokens from SMILES strings.
    
    # Parameters:
    `dataset`: List - List of SMILES strings
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
    
    for smiles in dataset:
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

def swap(dataset: List, attempts: int = 5, seed = 45) -> List[str]:
    """
    Swap two random tokens in SMILES strings.
    
    # Parameters:
    `dataset`: List - List of SMILES strings
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
    
    for smiles in dataset:
        tokens, non_charset_indices = atom_positions(smiles)
        swapped_smiles = set()
        
        if len(non_charset_indices) < 2:
            all_swapped_smiles.append(smiles)
            continue
        
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

def fusion(dataset: List, mask_ratio: float = 0.05, delete_ratio: float = 0.3, attempts: int = 5, seed = 45) -> List[str]:
    """
    Fusion of mask, delete and swap functions. For each SMILES string,
    randomly choose one of the three augmentation methods.
    
    # Parameters:
    `dataset`: List - List of SMILES strings
    `mask_ratio`: float - ratio of tokens to mask
    `delete_ratio`: float - ratio of tokens to delete
    `attempts`: int - number of augmentation attempts per SMILES
    `seed`: int or numpy.random.RandomState - random seed for reproducibility
    
    # Returns:
    `List[str]`: List of augmented SMILES strings
    """
    augmented_smiles = []
    
    if hasattr(seed, 'choice') and callable(seed.choice):
        rng = seed
    else:
        rng = np.random.RandomState(seed)
    
    if not dataset:
        raise ValueError("Empty dataset isn't a valid dataset.")
    
    for i, smiles in enumerate(dataset):
        # print(f"Processing SMILES {i+1}/{len(dataset)}: {smiles}")
        
        chosen = rng.choice(3, 1)[0]
        
        try:
            if chosen == 0:
                augmented = mask([smiles], mask_ratio=mask_ratio, attempts=attempts, seed=rng)
            elif chosen == 1:
                augmented = delete([smiles], delete_ratio=delete_ratio, attempts=attempts, seed=rng)
            else:
                augmented = swap([smiles], attempts=attempts, seed=rng)
            
            augmented_smiles.extend(augmented)
        
        except Exception as e:
            print(f"Error during augmentation of {smiles}: {str(e)}")
            raise ValueError(e)
    
    return augmented_smiles

def augment_dataset(dataset: pd.DataFrame, augmentation_methods: List[str], mask_ratio: float = 0.1, delete_ratio: float = 0.3, attempts: int = 10,
                    smiles_collumn: str = "SMILES", augment_percentage: float = 0.2, seed: int = 42):

    rng = np.random.RandomState(seed)

    augmented_df = dataset.copy()
    dataset['single_smiles'] = dataset['SMILES'].apply(lambda x: ast.literal_eval(x) if x.startswith('[') else [x])
    df_expanded = dataset.explode('single_smiles')


    target_new_rows = int(len(dataset) * augment_percentage)

    new_rows = []
    augmented_count = 0

    while augmented_count < target_new_rows:
        row_idx = rng.choice(len(df_expanded))
        original_idx = df_expanded.index[row_idx]
        row = df_expanded.iloc[row_idx].copy()

        smiles = row['single_smiles']

        try:
            if "mask" in augmentation_methods:
                augmented_smiles = mask([smiles], mask_ratio=mask_ratio, attempts=attempts, seed=rng)
            elif "delete" in augmentation_methods:
                augmented_smiles = delete([smiles], delete_ratio=delete_ratio, attempts=attempts, seed=rng)
            elif "swap" in augmentation_methods:
                augmented_smiles = swap([smiles], attempts=attempts, seed=rng)
            elif "fusion" in augmentation_methods:
                augmented_smiles = fusion([smiles], mask_ratio=mask_ratio, delete_ratio=delete_ratio, 
                                        attempts=attempts, seed=rng)
            elif "enumeration" in augmentation_methods:
                augmented_smiles = enumerateSmiles(smiles, num_randomizations=attempts)
            else:
                raise ValueError(f"Unknown augmentation methods: {augmentation_methods}")
    
            if augmented_smiles:
                for aug_smiles in augmented_smiles:
                    # Cria uma nova linha baseada na original
                    new_row = row.copy()
                    # Substitui o SMILES pelo aumentado
                    new_row[smiles_collumn] = aug_smiles
                    
                    # Substitui valores de propriedades por "-"
                    property_columns = [col for col in new_row.index if col.startswith('Property_')]
                    for prop_col in property_columns:
                        new_row[prop_col] = "-"
                    
                    # Adiciona a nova coluna com o índice da molécula original
                    new_row['parent_idx'] = original_idx
                    
                    new_rows.append(new_row)
                    augmented_count += 1
                
                # Para se já atingiu o número desejado
                if augmented_count >= target_new_rows:
                    break

        except Exception as e:
            print(f"Error augmenting SMILES {smiles}: {str(e)}")
            continue

    if new_rows:
        new_data = pd.DataFrame(new_rows)
        
        # Remove a coluna temporária
        if 'single_smiles' in new_data.columns:
            new_data = new_data.drop(columns=['single_smiles'])
        
        # Concatena com o dataset original
        augmented_df = pd.concat([augmented_df, new_data], ignore_index=True)
        
    return augmented_df

    
def random_select(dataset: pd.DataFrame, seed: int = 45) -> List[str]:
    """
    Shuffle dataset to augment a certain percentage of the data

    # Parameters:
    `dataset`: List - list of SMILES strings
    `seed`: int - random seed for reproducibility

    # Returns:
    `List[str]: Subset of data to augment`
    """
    
    if dataset.empty():
        raise ValueError("Dataset is empty. Load the dataset before trying to augment.")
    
    data_to_augment = dataset.sample(n=1, random_state=seed, replace=False)

    return data_to_augment