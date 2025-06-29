# What is Augchem?

Augchem is a comprehensive toolbox for chemical data augmentation developed in partnership with CINE and FAPESP. It provides state-of-the-art techniques for augmenting molecular data across multiple representations.

## Why use data augmentation for chemical data?

Machine Learning (ML) is gaining prominence in the discovery of new materials, complementing traditional methods by analyzing large volumes of data and identifying patterns. The effectiveness of ML models depends on data quality, making data augmentation techniques essential to improve model accuracy in materials chemistry. However, there is a lack of studies integrating these techniques in this field. This toolbox aims to address this gap with easy-to-use Python libraries for chemical data augmentation.

## Key Features

### 🔤 SMILES Augmentation
- **Token Manipulation**: Advanced parsing and modification of SMILES strings
- **Masking Techniques**: Replace molecular tokens with mask symbols for robust training
- **Deletion Strategies**: Remove random tokens to create structural variations
- **Atom Swapping**: Exchange atomic positions for diverse canonical representations
- **Fusion Methods**: Intelligently combine multiple augmentation techniques
- **Enumeration**: Generate non-canonical SMILES for increased diversity
- **Chemical Validation**: Built-in RDKit integration for molecular correctness
- **Property Preservation**: Maintain molecular properties during augmentation

### 🔗 Molecular Graph Augmentation
- **Edge Dropping**: Systematic removal of molecular bonds for structural variation
- **Node Dropping**: Atomic removal while preserving molecular validity
- **Feature Masking**: Node feature perturbation for robust representation learning
- **Edge Perturbation**: Dynamic bond addition and removal for chemical space exploration

### ⚡ Integration & Processing
- **PyTorch Geometric**: Native support for modern graph neural network workflows
- **Pandas Integration**: Seamless DataFrame processing for molecular datasets
- **Batch Operations**: Efficient processing of large molecular databases
- **RDKit Compatibility**: Chemical validation and property calculation
- **Reproducible Results**: Seed-based random state management

### 🛠️ Developer Experience
- **Simple API**: Intuitive interface through main Augmentator class
- **Flexible Parameters**: Customizable augmentation rates and methods
- **Quality Control**: Built-in validation and error handling
- **Memory Efficient**: Optimized tensor operations and data structures
- **GPU Support**: Acceleration for large-scale processing

## Target Applications

### 🔤 SMILES-Based Applications
- **Language Models**: Training chemical language models with augmented SMILES
- **Property Prediction**: Enhance datasets for molecular property regression
- **Drug Discovery**: Virtual compound generation and ADMET prediction
- **Chemical Space Exploration**: Systematic molecular diversity expansion
- **Text-Based ML**: Transformer and RNN training for chemical sequences

### 🔗 Graph-Based Applications
- **Graph Neural Networks**: Training data enhancement for chemical GNNs
- **Drug Discovery**: Molecular property prediction with enhanced datasets
- **Materials Science**: Crystal structure and property augmentation
- **Chemical Informatics**: Robust molecular representation learning
- **Structure-Activity Relationships**: SAR modeling with diverse molecular graphs

### 🔬 Research Applications  
- **Cheminformatics Research**: Systematic molecular dataset expansion
- **ML Benchmarking**: Standardized augmentation for fair model comparison
- **Data Scarcity**: Address limited training data in specialized chemical domains
- **Robustness Testing**: Evaluate model performance under molecular variations