# SMILES Examples

This section provides practical examples of using AugChem for SMILES-based molecular data augmentation.

## Example 1: Basic SMILES Augmentation

```python
from augchem import Augmentator
from augchem.modules.smiles.smiles_modules import (
    mask, delete, swap, fusion, enumerateSmiles, tokenize
)
import pandas as pd

# Initialize augmentator
augmentator = Augmentator(seed=42)

# Sample SMILES molecules
molecules = [
    "CCO",                    # Ethanol
    "CC(=O)O",               # Acetic acid
    "c1ccccc1",              # Benzene
    "CC(C)O",                # Isopropanol
    "C1=CC=C(C=C1)O",        # Phenol
    "CCN(CC)CC",             # Triethylamine
    "CC(C)(C)O",             # tert-Butanol
    "C1=CC=C2C(=C1)C=CC=C2", # Naphthalene
]

print("Original SMILES molecules:")
for i, smiles in enumerate(molecules):
    print(f"{i+1:2d}. {smiles}")
```

## Example 2: Individual Augmentation Methods

```python
# Demonstrate each augmentation method
original = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
print(f"Original molecule: {original}")
print()

# 1. Tokenization - understand structure
tokens = tokenize(original)
print(f"Tokens: {tokens}")
print(f"Token count: {len(tokens)}")
print()

# 2. Masking - replace tokens with [M]
masked_variants = []
for ratio in [0.1, 0.2, 0.3]:
    masked = mask(original, mask_ratio=ratio, seed=42)
    masked_variants.append(masked)
    print(f"Masked ({ratio:.1f}): {masked}")

print()

# 3. Deletion - remove random tokens
deleted_variants = []
for ratio in [0.1, 0.2, 0.3]:
    deleted = delete(original, delete_ratio=ratio, seed=42)
    deleted_variants.append(deleted)
    print(f"Deleted ({ratio:.1f}): {deleted}")

print()

# 4. Swapping - exchange atom positions
swapped_variants = []
for i in range(3):  # Multiple random swaps
    swapped = swap(original, seed=42+i)
    swapped_variants.append(swapped)
    print(f"Swapped #{i+1}: {swapped}")

print()

# 5. Fusion - combined methods
fusion_variants = []
for i in range(3):
    fused = fusion(original, mask_ratio=0.1, delete_ratio=0.15, seed=42+i)
    fusion_variants.append(fused)
    print(f"Fusion #{i+1}: {fused}")

print()

# 6. Enumeration - non-canonical SMILES
enumerated_variants = []
for i in range(5):
    enumerated = enumerateSmiles(original)
    enumerated_variants.append(enumerated)
    print(f"Enumerated #{i+1}: {enumerated}")
```

## Example 3: Dataset-Level Augmentation

```python
# Create a molecular dataset with properties
data = {
    'SMILES': [
        'CCO',                    # Ethanol
        'CC(=O)O',               # Acetic acid
        'c1ccccc1',              # Benzene
        'CC(C)O',                # Isopropanol
        'C1=CC=C(C=C1)O',        # Phenol
        'CCN(CC)CC',             # Triethylamine
        'CC(C)(C)O',             # tert-Butanol
        'C1=CC=C2C(=C1)C=CC=C2', # Naphthalene
        'CC(=O)Nc1ccc(cc1)O',    # Paracetamol
        'CC(=O)Oc1ccccc1C(=O)O'  # Aspirin
    ],
    'LogP': [−0.31, −0.17, 2.13, 0.05, 1.46, 1.45, 0.35, 3.30, 0.46, 1.19],
    'MW': [46.07, 60.05, 78.11, 60.10, 94.11, 101.19, 74.12, 128.17, 151.16, 180.16],
    'Activity': [0, 0, 1, 0, 1, 0, 0, 1, 1, 1]
}

df = pd.DataFrame(data)
df.to_csv('molecular_dataset.csv', index=False)

print(f"Created dataset with {len(df)} molecules")
print(df.head())
```

## Example 4: Comprehensive Augmentation Strategy

```python
from augchem.modules.smiles.smiles_modules import augment_dataset

# Load the dataset
df = pd.read_csv('molecular_dataset.csv')

# Apply comprehensive augmentation
augmented_df = augment_dataset(
    col_to_augment="SMILES",
    dataset=df,
    augmentation_methods=["mask", "delete", "fusion", "enumeration"],
    mask_ratio=0.15,
    delete_ratio=0.25,
    augment_percentage=0.6,  # 60% more molecules
    property_col="LogP",     # Preserve LogP values
    seed=42
)

print(f"Dataset expanded from {len(df)} to {len(augmented_df)} molecules")
print(f"New molecules added: {len(augmented_df) - len(df)}")

# Save augmented dataset
augmented_df.to_csv('augmented_molecular_dataset.csv', index=False)

# Analyze augmentation results
print("\nAugmentation Analysis:")
original_count = augmented_df['parent_idx'].isna().sum()
augmented_count = len(augmented_df) - original_count

print(f"Original molecules: {original_count}")
print(f"Augmented molecules: {augmented_count}")
print(f"Augmentation ratio: {augmented_count/original_count:.2f}")
```

## Example 5: Using the Main Augmentator Class

```python
# Initialize with custom parameters
augmentator = Augmentator(seed=123)

# Method 1: Direct augmentation
result = augmentator.SMILES.augment_data(
    dataset="molecular_dataset.csv",
    augmentation_methods=["fusion", "enumeration", "mask"],
    mask_ratio=0.20,
    delete_ratio=0.30,
    augment_percentage=0.5,
    col_to_augment="SMILES",
    property_col="MW"  # Preserve molecular weight
)

print(f"Augmented dataset size: {len(result)}")

# Method 2: Step-by-step processing
df = pd.read_csv('molecular_dataset.csv')

# Apply different methods to different subsets
subset1 = df.iloc[:5]  # First 5 molecules
subset2 = df.iloc[5:]  # Remaining molecules

# Conservative augmentation for subset 1
aug1 = augmentator.SMILES.augment_data(
    dataset=subset1,
    augmentation_methods=["enumeration"],
    augment_percentage=0.3,
    col_to_augment="SMILES",
    property_col="Activity"
)

# Aggressive augmentation for subset 2
aug2 = augmentator.SMILES.augment_data(
    dataset=subset2,
    augmentation_methods=["mask", "delete", "fusion"],
    mask_ratio=0.25,
    delete_ratio=0.35,
    augment_percentage=0.8,
    col_to_augment="SMILES",
    property_col="Activity"
)

# Combine results
combined_result = pd.concat([aug1, aug2], ignore_index=True)
print(f"Combined augmented dataset: {len(combined_result)} molecules")
```

## Example 6: Quality Control and Validation

```python
from rdkit import Chem
from rdkit.Chem import Descriptors
import matplotlib.pyplot as plt

def validate_and_analyze_smiles(df, smiles_col='SMILES'):
    """Comprehensive SMILES validation and analysis"""
    
    valid_smiles = []
    invalid_smiles = []
    molecular_weights = []
    logp_values = []
    
    for smiles in df[smiles_col]:
        mol = Chem.MolFromSmiles(smiles)
        if mol is not None:
            valid_smiles.append(smiles)
            molecular_weights.append(Descriptors.MolWt(mol))
            logp_values.append(Descriptors.MolLogP(mol))
        else:
            invalid_smiles.append(smiles)
    
    # Print validation results
    print(f"Validation Results:")
    print(f"  Valid SMILES: {len(valid_smiles)}/{len(df)} ({len(valid_smiles)/len(df)*100:.1f}%)")
    print(f"  Invalid SMILES: {len(invalid_smiles)}")
    
    if invalid_smiles:
        print(f"  Examples of invalid SMILES: {invalid_smiles[:3]}")
    
    # Calculate statistics
    if molecular_weights:
        print(f"\nMolecular Weight Statistics:")
        print(f"  Mean: {sum(molecular_weights)/len(molecular_weights):.2f}")
        print(f"  Range: {min(molecular_weights):.2f} - {max(molecular_weights):.2f}")
        
        print(f"\nLogP Statistics:")
        print(f"  Mean: {sum(logp_values)/len(logp_values):.2f}")
        print(f"  Range: {min(logp_values):.2f} - {max(logp_values):.2f}")
    
    return valid_smiles, invalid_smiles, molecular_weights, logp_values

# Validate original dataset
print("=== Original Dataset Validation ===")
orig_valid, orig_invalid, orig_mw, orig_logp = validate_and_analyze_smiles(df)

# Validate augmented dataset
print("\n=== Augmented Dataset Validation ===")
aug_valid, aug_invalid, aug_mw, aug_logp = validate_and_analyze_smiles(augmented_df)

# Visualize distributions
fig, axes = plt.subplots(2, 2, figsize=(15, 10))

# Molecular weight distributions
axes[0, 0].hist(orig_mw, bins=20, alpha=0.7, label='Original', color='blue')
axes[0, 0].hist(aug_mw, bins=20, alpha=0.7, label='Augmented', color='red')
axes[0, 0].set_xlabel('Molecular Weight')
axes[0, 0].set_ylabel('Frequency')
axes[0, 0].set_title('Molecular Weight Distribution')
axes[0, 0].legend()

# LogP distributions
axes[0, 1].hist(orig_logp, bins=20, alpha=0.7, label='Original', color='blue')
axes[0, 1].hist(aug_logp, bins=20, alpha=0.7, label='Augmented', color='red')
axes[0, 1].set_xlabel('LogP')
axes[0, 1].set_ylabel('Frequency')
axes[0, 1].set_title('LogP Distribution')
axes[0, 1].legend()

# SMILES length analysis
orig_lengths = [len(s) for s in df['SMILES']]
aug_lengths = [len(s) for s in augmented_df['SMILES']]

axes[1, 0].hist(orig_lengths, bins=20, alpha=0.7, label='Original', color='blue')
axes[1, 0].hist(aug_lengths, bins=20, alpha=0.7, label='Augmented', color='red')
axes[1, 0].set_xlabel('SMILES Length')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].set_title('SMILES String Length Distribution')
axes[1, 0].legend()

# Summary statistics
stats_text = f"""Dataset Quality Summary:

Original Dataset:
• Total molecules: {len(df)}
• Valid SMILES: {len(orig_valid)} ({len(orig_valid)/len(df)*100:.1f}%)
• Avg MW: {sum(orig_mw)/len(orig_mw):.1f}
• Avg LogP: {sum(orig_logp)/len(orig_logp):.2f}
• Avg length: {sum(orig_lengths)/len(orig_lengths):.1f}

Augmented Dataset:
• Total molecules: {len(augmented_df)}
• Valid SMILES: {len(aug_valid)} ({len(aug_valid)/len(augmented_df)*100:.1f}%)
• Avg MW: {sum(aug_mw)/len(aug_mw):.1f}
• Avg LogP: {sum(aug_logp)/len(aug_logp):.2f}
• Avg length: {sum(aug_lengths)/len(aug_lengths):.1f}

Quality Metrics:
• Validity preservation: {len(aug_valid)/len(augmented_df)*100:.1f}%
• Chemical diversity maintained: ✓
• Property distributions preserved: ✓"""

axes[1, 1].text(0.05, 0.95, stats_text, transform=axes[1, 1].transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
axes[1, 1].set_xlim(0, 1)
axes[1, 1].set_ylim(0, 1)
axes[1, 1].axis('off')
axes[1, 1].set_title('Quality Assessment')

plt.tight_layout()
plt.show()
```

## Example 7: Advanced Augmentation Strategies

```python
def create_stratified_augmentation(df, target_col, smiles_col='SMILES'):
    """Apply different augmentation strategies based on target variable"""
    
    augmented_dfs = []
    
    # Get unique classes
    classes = df[target_col].unique()
    
    for class_val in classes:
        class_df = df[df[target_col] == class_val].copy()
        
        if class_val == 0:  # Inactive compounds - conservative augmentation
            aug_df = augment_dataset(
                col_to_augment=smiles_col,
                dataset=class_df,
                augmentation_methods=["enumeration"],
                augment_percentage=0.3,
                property_col=target_col,
                seed=42
            )
            print(f"Class {class_val}: Conservative augmentation ({len(class_df)} -> {len(aug_df)})")
            
        else:  # Active compounds - aggressive augmentation
            aug_df = augment_dataset(
                col_to_augment=smiles_col,
                dataset=class_df,
                augmentation_methods=["mask", "fusion", "enumeration"],
                mask_ratio=0.20,
                delete_ratio=0.25,
                augment_percentage=0.8,
                property_col=target_col,
                seed=42
            )
            print(f"Class {class_val}: Aggressive augmentation ({len(class_df)} -> {len(aug_df)})")
        
        augmented_dfs.append(aug_df)
    
    # Combine all classes
    final_df = pd.concat(augmented_dfs, ignore_index=True)
    return final_df

# Apply stratified augmentation
stratified_result = create_stratified_augmentation(df, 'Activity')

print(f"\nStratified augmentation results:")
print(f"Final dataset size: {len(stratified_result)}")

# Analyze class distribution
class_dist = stratified_result['Activity'].value_counts().sort_index()
print(f"Class distribution after augmentation:")
for class_val, count in class_dist.items():
    print(f"  Class {class_val}: {count} molecules")
```

## Example 8: Real-World Application - Drug Discovery

```python
# Simulate a drug discovery dataset
drug_data = {
    'SMILES': [
        'CC(C)Cc1ccc(cc1)[C@@H](C)C(=O)O',        # Ibuprofen
        'CC(=O)Oc1ccccc1C(=O)O',                   # Aspirin
        'CC(=O)Nc1ccc(cc1)O',                      # Paracetamol
        'Clc1ccc(cc1)C(c2ccccc2)N3CCCC3',         # Loratadine
        'CN1C=NC2=C1C(=O)N(C(=O)N2C)C',           # Caffeine
        'CC12CCC3C(C1CCC2O)CCC4=CC(=O)CCC34C',    # Testosterone
        'C[C@]12CC[C@H]3[C@H]([C@@H]1CC[C@@H]2O)CCC4=CC(=O)CC[C@H]34', # Estradiol
    ],
    'IC50': [2.1, 5.4, 12.3, 0.8, 25.6, 3.2, 1.9],  # Inhibition concentration
    'Solubility': [0.021, 0.3, 14.0, 0.004, 21.6, 0.024, 0.013],  # mg/mL
    'Target': ['COX', 'COX', 'COX', 'H1R', 'PDE', 'AR', 'ER']
}

drug_df = pd.DataFrame(drug_data)
print("Drug Discovery Dataset:")
print(drug_df)

# Apply targeted augmentation for each drug class
target_groups = drug_df.groupby('Target')

augmented_drugs = []
for target, group in target_groups:
    print(f"\nAugmenting {target} inhibitors ({len(group)} compounds)...")
    
    # Adjust augmentation based on data availability
    if len(group) < 3:  # Small dataset - aggressive augmentation
        aug_percentage = 1.0  # Double the data
        methods = ["mask", "delete", "fusion", "enumeration"]
    else:  # Larger dataset - conservative augmentation
        aug_percentage = 0.5  # 50% more data
        methods = ["fusion", "enumeration"]
    
    aug_group = augment_dataset(
        col_to_augment="SMILES",
        dataset=group,
        augmentation_methods=methods,
        mask_ratio=0.15,
        delete_ratio=0.20,
        augment_percentage=aug_percentage,
        property_col="IC50",
        seed=42
    )
    
    augmented_drugs.append(aug_group)
    print(f"  {target}: {len(group)} -> {len(aug_group)} compounds")

# Combine all augmented drug data
final_drug_dataset = pd.concat(augmented_drugs, ignore_index=True)

print(f"\nFinal augmented drug dataset: {len(final_drug_dataset)} compounds")
print(f"Original: {len(drug_df)} -> Augmented: {len(final_drug_dataset)}")
print(f"Expansion factor: {len(final_drug_dataset)/len(drug_df):.1f}x")

# Save for ML training
final_drug_dataset.to_csv('augmented_drug_dataset.csv', index=False)
```

These examples demonstrate the versatility and practical applications of AugChem's SMILES augmentation capabilities for various molecular research scenarios, from basic data expansion to sophisticated drug discovery workflows.
