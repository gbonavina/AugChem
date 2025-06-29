# Augchem

Welcome to the documentation for **Augchem**, a Python toolbox for chemical data augmentation, developed in partnership with FAPESP and CINE.

## Overview

**Augchem** provides comprehensive tools for chemical data augmentation across multiple molecular representations:

- **🔤 SMILES**: String-based molecular representation augmentation with advanced text manipulation
- **🔗 Graphs**: PyTorch Geometric-based molecular graph augmentation with structural modifications
- **🧬 InChI**: International Chemical Identifier augmentation for standardized representations

## Features

### 🔤 SMILES Augmentation
- **Masking**: Replace molecular tokens with mask symbols for robust training
- **Deletion**: Remove random tokens to create structural variations  
- **Swapping**: Exchange atom positions for diverse canonical forms
- **Fusion**: Combine multiple augmentation techniques intelligently
- **Enumeration**: Generate non-canonical SMILES representations
- **Dataset Processing**: Batch augmentation of molecular datasets with property preservation
- **Quality Control**: Built-in validation using RDKit for chemical correctness

### 🔗 Graph Augmentation
- **Edge Dropping**: Remove bidirectional edges to create structural variations
- **Node Dropping**: Remove nodes while maintaining graph integrity
- **Feature Masking**: Mask node features for robust representation learning
- **Edge Perturbation**: Add and remove edges to explore chemical space
- **Batch Processing**: Efficient processing using PyTorch Geometric DataLoaders
- **Advanced Analytics**: Comprehensive graph statistics and visualization tools

### ⚡ Integration & Compatibility
- **PyTorch Geometric**: Native support for graph neural networks
- **RDKit Integration**: Chemical validation and property calculation
- **Pandas Support**: Seamless DataFrame processing for datasets
- **Reproducible Results**: Seed-based random state management

## Key Features

### 🎯 SMILES Processing
- **Token-Level Manipulation**: Intelligent parsing and modification of SMILES strings
- **Chemical Validity**: Automatic validation to ensure augmented molecules remain valid
- **Property Preservation**: Maintain molecular properties during augmentation
- **Flexible Parameters**: Customizable masking, deletion, and fusion ratios
- **Batch Operations**: Process entire molecular datasets efficiently

### 🔗 Graph Processing  
- **Multi-Technique Augmentation**: Combine edge, node, and feature modifications
- **Self-Loop Detection**: Automatic cleanup for graph neural network compatibility
- **Batch Collation**: Optimized for PyTorch Geometric DataLoaders
- **Quality Metrics**: Built-in graph validation and statistics

### 🛠️ Developer Experience
- **Simple API**: Intuitive interface for both beginners and experts
- **Comprehensive Documentation**: Detailed tutorials and examples
- **Extensible Design**: Easy to add custom augmentation techniques
- **Production Ready**: Tested and optimized for research and industry use

## Quick Start

### SMILES Augmentation
```python
from augchem import Augmentator

# Initialize augmentator
augmentator = Augmentator(seed=42)

# Augment SMILES dataset
result = augmentator.SMILES.augment_data(
    dataset="molecules.csv",
    augmentation_methods=["mask", "fusion", "enumeration"],
    augment_percentage=0.3,
    col_to_augment="SMILES"
)
```

### Graph Augmentation
```python
from augchem.modules.graph.graphs_modules import augment_dataset

# Apply multiple augmentation techniques
augmented_graphs = augment_dataset(
    graphs=your_graphs,
    augmentation_methods=['edge_drop', 'node_drop', 'feature_mask'],
    augment_percentage=0.2
)
```

## 🚀 Getting Started

Ready to enhance your molecular datasets? Choose your path:

- **📚 [Tutorials](tutorial.md)** - Step-by-step learning guides
- **💡 [Examples](examples.md)** - Practical applications and use cases  
- **📖 [API Reference](reference/augchem.md)** - Complete technical documentation

Explore the comprehensive documentation to master both SMILES and graph augmentation techniques!