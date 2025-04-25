from typing import Tuple, List, Optional
from rdkit import Chem
from rdkit import RDLogger
import numpy as np
import pandas as pd
import re

# disable rdkit warnings
RDLogger.DisableLog('rdApp.*')


def atom_positions(smiles: str) -> Tuple[List[str], List[int]]:
    """
    Extracts individual characters from a SMILES string and identifies indices of atoms.
    
    This function tokenizes a SMILES string into individual characters and identifies
    positions of actual atoms by excluding special characters like brackets, 
    parentheses, bonds, digits, etc.
    
    Parameters
    ----------
    `smiles` : str
        A valid SMILES string representation of a molecule
    
    Returns
    -------
    `Tuple[List[str]`, `List[int]]`
        A tuple containing:
        - List of individual characters from the SMILES string
        - List of indices where non-special characters (atoms) are located
    
    Examples
    --------
    >>> atom_positions("CC(=O)O") = (['C', 'C', '(', '=', 'O', ')', 'O'], [0, 1, 4, 6])
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
    Tokenizes a SMILES string using a regular expression pattern.
    
    Splits a SMILES string into chemically meaningful tokens according to a
    predefined regex pattern. This tokenization preserves atom types, bonds,
    stereochemistry, and other structural features.
    
    Parameters
    ----------
    `smiles` : str
        A valid SMILES string representation of a molecule
    
    Returns
    -------
    `List[str]`
        A list of chemical tokens extracted from the SMILES string
    
    Examples
    --------
    >>> tokenize("CC(=O)O") = ['C', 'C', '(', '=', 'O', ')', 'O']
    
    >>> tokenize("C1=CC=CC=C1") = ['C', '1', '=', 'C', 'C', '=', 'C', 'C', '=', 'C', '1']
    """

    SMI_REGEX_PATTERN = r"""(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|\.|=|#|-|\+|\\|\/|:|~|@|\?|>>?|\*|\$|\%[0-9]{2}|[0-9])"""
    regex = re.compile(SMI_REGEX_PATTERN)

    tokens = [token for token in regex.findall(smiles)]
    return tokens

def enumerateSmiles(smiles: str) -> Optional[str]:
    """
    Generates a valid non-canonical SMILES representation of the input molecule.
    
    Creates an alternative, but chemically equivalent SMILES string by randomizing
    the atom ordering while preserving the molecular structure. Returns None if 
    the generation fails or produces an invalid SMILES.
    
    Parameters
    ----------
    `smiles` : str
        A valid SMILES string representation of a molecule
    
    Returns
    -------
    `Optional[str]`
        A new, valid SMILES string with randomized atom ordering, or None if generation fails
    
    Raises
    ------
    ValueError
        If the input SMILES string is invalid
        
    Examples
    --------
    >>> enumerateSmiles("CC(=O)O") = 'OC(C)=O'
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
    Replaces random tokens in a SMILES string with a masking token '[M]'.
    
    Tokenizes the SMILES string and randomly replaces a specified fraction of tokens
    with a mask token. Useful for creating partially obscured molecular representations
    for machine learning applications.
    
    Parameters
    ----------
    `smiles` : str
        A valid SMILES string representation of a molecule

    `mask_ratio` : float, default=0.5
        Fraction of tokens to replace with mask tokens (0.0 to 1.0)

    `seed` : int or numpy.random.RandomState, default=45
        Random seed or random number generator for reproducibility
    
    Returns
    -------
    `str`
        SMILES string with selected tokens replaced by '[M]'
    
    Examples
    --------
    >>> mask("CC(=O)O", mask_ratio=0.4, seed=42) = 'C[M](=O)[M]'
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
    Removes random tokens from a SMILES string.
    
    Tokenizes the SMILES string and randomly deletes a specified fraction of tokens.
    This creates an incomplete representation that can be used for data augmentation
    or model robustness testing.
    
    Parameters
    ----------
    `smiles` : str
        A valid SMILES string representation of a molecule

    `delete_ratio` : float, default=0.3
        Fraction of tokens to delete (0.0 to 1.0)

    `seed` : int or numpy.random.RandomState, default=45
        Random seed or random number generator for reproducibility
    
    Returns
    -------
    `str`
        SMILES string with selected tokens removed
    
    Examples
    --------
    >>> delete("CC(=O)O", delete_ratio=0.3, seed=42) = 'C(=O)O'
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
    Exchanges two random atom tokens within a SMILES string.
    
    Identifies non-special character positions in the SMILES string and swaps
    two randomly selected atoms. This preserves the token count but alters
    the molecular structure.
    
    Parameters
    ----------
    `smiles` : str
        A valid SMILES string representation of a molecule
    `seed` : int or numpy.random.RandomState, default=45
        Random seed or random number generator for reproducibility
    
    Returns
    -------
    str
        SMILES string with two atoms swapped
    
    Examples
    --------
    >>> swap("CC(=O)O", seed=42) = 'OC(=O)C'
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
    Applies one randomly selected augmentation method to a SMILES string.
    
    Randomly chooses between masking, deletion, or swapping transformations and
    applies it to the input SMILES. This provides a diverse set of augmentation
    possibilities with a single function call.
    
    Parameters
    ----------
    `smiles` : str
        A valid SMILES string representation of a molecule

    `mask_ratio` : float, default=0.05
        Fraction of tokens to mask if masking is selected (0.0 to 1.0)

    `delete_ratio` : float, default=0.3
        Fraction of tokens to delete if deletion is selected (0.0 to 1.0)

    `seed` : int or numpy.random.RandomState, default=45
        Random seed or random number generator for reproducibility
    
    Returns
    -------
    `str`
        Augmented SMILES string
    
    Raises
    ------
    ValueError
        If input SMILES is empty or if augmentation fails
    
    Examples
    --------
    >>> fusion("CC(=O)O", seed=42) = 'CC(O)='
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
    """
    Applies selected augmentation methods to SMILES strings in a dataset.
    
    Generates augmented variants of molecular SMILES strings using specified methods
    and adds them to the dataset. Tracks relationships between original and augmented
    molecules using parent indices.
    
    Parameters
    ----------
    `col_to_augment` : str
        Column name containing SMILES strings to augment

    `dataset` : pd.DataFrame
        DataFrame containing molecular data with SMILES strings

    `augmentation_methods` : List[str]
        List of methods to apply. Valid options: "mask", "delete", "swap", "fusion", "enumeration"
    
    `mask_ratio` : float, default=0.1
        Fraction of tokens to mask when using mask augmentation
    
    `property_col` : str, optional
        Column name containing property values to preserve in augmented data
    
    `delete_ratio` : float, default=0.3
        Fraction of tokens to delete when using delete augmentation
    
    `augment_percentage` : float, default=0.2
        Target size of augmented dataset as a fraction of original dataset size
    
    `seed` : int, default=42
        Random seed for reproducibility
    
    Returns
    -------
    `pd.DataFrame`
        Original dataset with augmented molecules appended, including a 'parent_idx'
        column that references original molecule indices
    
    Raises
    ------
    ValueError
        If input data is not in SMILES format or an unknown augmentation method is specified
    
    Notes
    -----
    Property columns with names starting with "Property_" will be set to "-" in augmented rows.
    """

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
    
