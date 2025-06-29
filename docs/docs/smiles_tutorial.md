# SMILES Augmentation Tutorial

This tutorial demonstrates how to use AugChem's SMILES augmentation capabilities for molecular data enhancement.

## Prerequisites

```bash
pip install augchem rdkit pandas
```

## Basic Setup

```python
from augchem import Augmentator
import pandas as pd

# Initialize the augmentator with a seed for reproducibility
augmentator = Augmentator(seed=42)
```

## Loading and Preparing SMILES Data

```python
import pandas as pd

# Create a sample dataset
data = {
    'SMILES': [
        'CC(=O)O',           # Acetic acid
        'CCO',               # Ethanol  
        'C1=CC=CC=C1',       # Benzene
        'CC(C)O',            # Isopropanol
        'C1=CC=C(C=C1)O'     # Phenol
    ],
    'Property_0': [0.45, 1.23, -0.87, 0.62, -0.34]
}

df = pd.DataFrame(data)
df.to_csv('molecules.csv', index=False)
print("Sample dataset created!")
```

## Individual SMILES Augmentation Methods

```python
from augchem.modules.smiles.smiles_modules import (
    mask, delete, swap, fusion, enumerateSmiles, tokenize
)

original_smiles = "CC(=O)O"  # Acetic acid
print(f"Original SMILES: {original_smiles}")

# 1. Tokenization - understand SMILES structure
tokens = tokenize(original_smiles)
print(f"Tokens: {tokens}")

# 2. Masking - replace tokens with [M]
masked = mask(original_smiles, mask_ratio=0.3, seed=42)
print(f"Masked: {masked}")

# 3. Deletion - remove random tokens
deleted = delete(original_smiles, delete_ratio=0.2, seed=42)
print(f"Deleted: {deleted}")

# 4. Swapping - exchange atom positions
swapped = swap(original_smiles, seed=42)
print(f"Swapped: {swapped}")

# 5. Fusion - randomly apply mask/delete/swap
fused = fusion(original_smiles, mask_ratio=0.1, delete_ratio=0.2, seed=42)
print(f"Fused: {fused}")

# 6. Enumeration - generate non-canonical SMILES
enumerated = enumerateSmiles(original_smiles)
print(f"Enumerated: {enumerated}")
```

## Dataset-Level SMILES Augmentation

```python
from augchem.modules.smiles.smiles_modules import augment_dataset

# Load your dataset
df = pd.read_csv('molecules.csv')

# Apply augmentation using individual function
augmented_df = augment_dataset(
    col_to_augment="SMILES",
    dataset=df,
    augmentation_methods=["mask", "delete", "fusion", "enumeration"],
    mask_ratio=0.1,
    delete_ratio=0.3,
    augment_percentage=0.4,  # 40% more molecules
    property_col="Property_0",
    seed=42
)

print(f"Original: {len(df)} molecules")
print(f"Augmented: {len(augmented_df)} molecules")
print(f"New molecules: {len(augmented_df) - len(df)}")
```

## Using the Main Augmentator Class (Recommended)

```python
# Using the main Augmentator class
augmentator = Augmentator(seed=42)

# Augment SMILES data
result = augmentator.SMILES.augment_data(
    dataset="molecules.csv",
    augmentation_methods=["fusion", "enumeration", "mask"],
    mask_ratio=0.15,
    delete_ratio=0.25,
    augment_percentage=0.3,
    col_to_augment="SMILES",
    property_col="Property_0"
)

print(f"Augmentation complete! Dataset saved as 'Augmented_molecules.csv'")
print(f"Final dataset size: {len(result)} molecules")
```

## SMILES Augmentation Methods Summary

| Method | Description | Parameters | Best For |
|--------|-------------|------------|----------|
| **mask** | Replace tokens with '[M]' | `mask_ratio` | Language modeling |
| **delete** | Remove random tokens | `delete_ratio` | Robustness testing |
| **swap** | Exchange atom positions | None | Structural variation |
| **fusion** | Random method selection | `mask_ratio`, `delete_ratio` | Diverse augmentation |
| **enumeration** | Non-canonical SMILES | None | Canonical diversity |

## Understanding SMILES Augmentation Results

```python
# Load and analyze augmentation results
result = pd.read_csv('Augmented_molecules.csv')

# Check original vs augmented
original_count = result['parent_idx'].isna().sum()
augmented_count = len(result) - original_count

print(f"Original molecules: {original_count}")
print(f"Augmented molecules: {augmented_count}")

# Look at augmentation examples
augmented_only = result[result['parent_idx'].notna()]
print("\nAugmentation examples:")
for i in range(min(5, len(augmented_only))):
    row = augmented_only.iloc[i]
    parent_idx = int(row['parent_idx'])
    original = result.iloc[parent_idx]['SMILES']
    augmented = row['SMILES']
    print(f"Original: {original} → Augmented: {augmented}")
```

## SMILES Quality Control

```python
from rdkit import Chem

def validate_smiles(smiles_list):
    """Validate SMILES strings using RDKit"""
    valid_count = 0
    invalid_smiles = []
    
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_count += 1
        else:
            invalid_smiles.append(smiles)
    
    return valid_count, invalid_smiles

# Validate augmented SMILES
smiles_list = result['SMILES'].tolist()
valid_count, invalid = validate_smiles(smiles_list)

print(f"Valid SMILES: {valid_count}/{len(smiles_list)}")
print(f"Invalid SMILES: {len(invalid)}")
if invalid:
    print("Examples of invalid SMILES:", invalid[:3])
```

## Parameter Optimization

```python
def find_optimal_smiles_parameters():
    """Find optimal SMILES augmentation parameters"""
    
    test_params = [
        {'mask_ratio': 0.1, 'delete_ratio': 0.2, 'augment_percentage': 0.2},
        {'mask_ratio': 0.15, 'delete_ratio': 0.3, 'augment_percentage': 0.3},
        {'mask_ratio': 0.2, 'delete_ratio': 0.25, 'augment_percentage': 0.4},
    ]
    
    results = []
    
    for params in test_params:
        result = augmentator.SMILES.augment_data(
            dataset="molecules.csv",
            **params,
            augmentation_methods=["fusion", "mask"]
        )
        
        # Calculate validity ratio
        valid_count, invalid = validate_smiles(result['SMILES'].tolist())
        valid_ratio = valid_count / len(result)
        
        results.append({
            **params,
            'total_molecules': len(result),
            'valid_ratio': valid_ratio
        })
    
    return pd.DataFrame(results)

# Run optimization
optimization_results = find_optimal_smiles_parameters()
print(optimization_results)
```

## Large Dataset Processing

```python
def process_large_dataset(csv_path, chunk_size=1000):
    """Process large datasets in chunks"""
    
    # Read dataset info
    with open(csv_path) as f:
        total_rows = sum(1 for line in f) - 1  # subtract header
    
    augmented_chunks = []
    
    for chunk_start in range(0, total_rows, chunk_size):
        # Read chunk
        chunk = pd.read_csv(csv_path, skiprows=range(1, chunk_start+1), nrows=chunk_size)
        
        # Augment chunk
        augmented_chunk = augmentator.SMILES.augment_data(
            dataset=chunk,
            augmentation_methods=["fusion"],
            augment_percentage=0.1
        )
        
        augmented_chunks.append(augmented_chunk)
        print(f"Processed chunk {chunk_start//chunk_size + 1}")
    
    # Combine results
    final_result = pd.concat(augmented_chunks, ignore_index=True)
    return final_result

# Example usage for large datasets
# large_result = process_large_dataset("large_molecules.csv", chunk_size=500)
```

## Best Practices for SMILES Augmentation

### 1. **Validation First**
Always validate SMILES strings before and after augmentation:

```python
# Validate input
valid_input = [s for s in original_smiles if validate_smiles([s])[0] > 0]
print(f"Valid input SMILES: {len(valid_input)}/{len(original_smiles)}")
```

### 2. **Conservative Parameters**
Start with low augmentation ratios:

```python
# Recommended starting parameters
CONSERVATIVE_PARAMS = {
    'mask_ratio': 0.1,
    'delete_ratio': 0.2,
    'augment_percentage': 0.2
}
```

### 3. **Chemical Diversity Monitoring**
Check that augmentation preserves chemical diversity:

```python
from rdkit import Chem

def check_diversity(smiles_list):
    canonical_set = set()
    for smiles in smiles_list:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            canonical = Chem.MolToSmiles(mol)
            canonical_set.add(canonical)
    return len(canonical_set)

orig_diversity = check_diversity(original_smiles)
aug_diversity = check_diversity(augmented_smiles)
print(f"Diversity preserved: {aug_diversity >= orig_diversity}")
```

### 4. **Reproducible Augmentation**
Always use seeds for reproducible results:

```python
# Reproducible augmentation
EXPERIMENT_SEED = 42
augmentator = Augmentator(seed=EXPERIMENT_SEED)

augmented_1 = augmentator.SMILES.augment_data("molecules.csv", seed=EXPERIMENT_SEED)
augmented_2 = augmentator.SMILES.augment_data("molecules.csv", seed=EXPERIMENT_SEED)
# Results will be identical
```

## Common Issues and Solutions

### Invalid SMILES Generation
```python
# Filter out invalid SMILES after augmentation
def filter_valid_smiles(df):
    valid_mask = []
    for smiles in df['SMILES']:
        mol = Chem.MolFromSmiles(smiles)
        valid_mask.append(mol is not None)
    
    return df[valid_mask].reset_index(drop=True)

# Apply filter
clean_result = filter_valid_smiles(result)
print(f"Filtered: {len(result)} → {len(clean_result)} valid molecules")
```

### Property Column Handling
```python
# Ensure property columns are correctly preserved
result_with_properties = augmentator.SMILES.augment_data(
    dataset="molecules.csv",
    property_col="Property_0",  # Specify property column
    augmentation_methods=["enumeration"]  # Use safe methods
)

# Check property preservation
original_props = df['Property_0'].tolist()
augmented_props = result_with_properties['Property_0'].tolist()
print(f"Properties preserved: {len(set(original_props))} unique values")
```

## Integration with Machine Learning

```python
from sklearn.model_selection import train_test_split

# Prepare augmented dataset for ML
augmented_smiles = result['SMILES'].tolist()
augmented_labels = result['Property_0'].tolist()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    augmented_smiles, augmented_labels, 
    test_size=0.2, random_state=42
)

print(f"Training set: {len(X_train)} molecules")
print(f"Test set: {len(X_test)} molecules")

# Ready for molecular descriptors calculation and ML training
```

## Next Steps

After completing this SMILES tutorial:

- ✅ **Understand all SMILES augmentation methods**
- ✅ **Apply quality control and validation**
- ✅ **Optimize parameters for your datasets**
- ✅ **Handle large datasets efficiently**
- ✅ **Integrate with ML pipelines**

### See Also

- [Graph Augmentation Tutorial](graph_tutorial.md) - Learn graph-based augmentation
- [SMILES Methods API](reference/smiles_methods.md) - Complete method documentation
- [Examples](examples.md) - Real-world applications
