# Examples

This section provides practical examples of using AugChem for molecular data augmentation. Choose the appropriate section based on your data type and augmentation needs.

## 🧪 Quick Overview

AugChem supports two main types of molecular data augmentation:

- **🔤 SMILES Augmentation**: String-based molecular representation augmentation
- **🔗 Graph Augmentation**: Graph neural network-ready molecular graph augmentation

## 📋 Available Example Collections

### [SMILES Examples](smiles_examples.md)
Comprehensive examples for SMILES-based molecular augmentation including:
- Basic SMILES manipulation techniques
- Dataset-level augmentation strategies  
- Quality control and validation
- Real-world drug discovery applications
- Integration with cheminformatics workflows

### [Graph Examples](graph_examples.md)  
Detailed examples for graph-based molecular augmentation including:
- PyTorch Geometric integration
- Individual augmentation techniques
- Machine learning pipeline integration
- Comparative analysis and visualization
- Advanced pharmaceutical applications

## 🚀 Getting Started

If you're new to AugChem, we recommend:

1. **Start with Prerequisites**: Install required packages
2. **Choose Your Data Type**: SMILES strings or molecular graphs
3. **Follow Relevant Examples**: Pick examples that match your use case
4. **Experiment**: Modify parameters to suit your specific needs

### Prerequisites

```bash
pip install augchem torch torch-geometric rdkit pandas matplotlib
```

### Basic Usage Pattern

```python
from augchem import Augmentator

# Initialize with reproducible seed
augmentator = Augmentator(seed=42)

# For SMILES data
smiles_result = augmentator.SMILES.augment_data(
    dataset="your_data.csv",
    augmentation_methods=["fusion", "enumeration"],
    augment_percentage=0.5
)

# For Graph data (when available)
# graph_result = augmentator.Graph.augment_dataset(...)
```

## 🎯 Example Categories

### Beginner Examples
- Basic augmentation setup
- Single molecule processing
- Simple dataset expansion

### Intermediate Examples
- Parameter optimization
- Quality control implementation
- Integration with ML pipelines

### Advanced Examples
- Custom augmentation strategies
- Large-scale processing
- Research-grade applications

## 💡 Tips for Using Examples

1. **Modify Parameters**: Adjust augmentation rates based on your data
2. **Validate Results**: Always check output quality
3. **Set Seeds**: Use random seeds for reproducible experiments
4. **Start Small**: Test with small datasets first
5. **Monitor Performance**: Track augmentation impact on model performance

## 🔬 Real-World Applications

Our examples cover scenarios from:
- **Academic Research**: Dataset expansion for publications
- **Drug Discovery**: Virtual compound generation
- **Chemical Informatics**: Property prediction enhancement
- **Materials Science**: Novel structure exploration

---

## 📖 Additional Resources

- [SMILES Tutorial](smiles_tutorial.md) - Step-by-step learning guide
- [Graph Tutorial](graph_tutorial.md) - Comprehensive graph augmentation guide  
- [API Reference](reference/) - Complete function documentation

---

**Ready to augment your molecular data? Choose your examples and start exploring! 🧬✨**
 