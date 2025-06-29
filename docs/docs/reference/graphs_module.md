# Graph Module

The Graph Module provides comprehensive tools for molecular graph data augmentation using PyTorch Geometric. This module includes advanced augmentation techniques specifically designed for chemical molecular graphs.

## Module Overview

::: augchem.modules.graph
    options:
      show_source: false
      heading_level: 3

## Key Components

### Graphs Modules
The core functionality for graph augmentation techniques:

::: augchem.modules.graph.graphs_modules
    options:
      show_source: false
      heading_level: 4

## Available Augmentation Techniques

| Technique | Description | Use Case |
|-----------|-------------|----------|
| **Edge Dropping** | Removes complete bidirectional edges | Structural perturbation, robustness testing |
| **Node Dropping** | Removes nodes and associated edges | Graph topology variation, missing data simulation |
| **Feature Masking** | Masks node features with -inf values | Feature robustness, attention mechanism training |
| **Edge Perturbation** | Adds and removes edges simultaneously | Chemical space exploration, bond variation |

## Integration Features

### PyTorch Geometric Compatibility
- Native support for `torch_geometric.data.Data` objects
- Seamless integration with PyTorch Geometric DataLoaders
- Optimized for graph neural network training pipelines

### Batch Processing
- Efficient processing of multiple graphs
- GPU acceleration support
- Memory-optimized operations

### Quality Assurance
- Graph integrity validation
- Self-loop detection and removal
- Consistent edge attribute handling

## Example Workflow

```python
from augchem.modules.graph.graphs_modules import augment_dataset

# Define your molecular graphs
molecular_graphs = load_your_molecular_graphs()

# Apply comprehensive augmentation
augmented_dataset = augment_dataset(
    graphs=molecular_graphs,
    augmentation_methods=['edge_drop', 'node_drop', 'feature_mask'],
    augment_percentage=0.25,
    seed=42
)

# Use in your machine learning pipeline
from torch_geometric.loader import DataLoader
loader = DataLoader(augmented_dataset, batch_size=32, shuffle=True)
```

## Performance Considerations

- **Memory Usage**: All operations create graph clones to preserve originals
- **GPU Support**: Full tensor operation compatibility with CUDA
- **Scalability**: Optimized for large molecular datasets
- **Reproducibility**: Deterministic results with seed control

For detailed function documentation and examples, see the [Graph Methods](graphs_methods.md) section.