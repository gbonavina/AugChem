# Graph Augmentation Methods

This section covers the advanced graph augmentation techniques available in Augchem, designed specifically for molecular graphs using PyTorch Geometric.

## Core Augmentation Functions

### Edge Dropping
::: augchem.modules.graph.graphs_modules.edge_dropping

### Node Dropping  
::: augchem.modules.graph.graphs_modules.node_dropping

### Feature Masking
::: augchem.modules.graph.graphs_modules.feature_masking

### Edge Perturbation
::: augchem.modules.graph.graphs_modules.edge_perturbation

### Dataset Augmentation
::: augchem.modules.graph.graphs_modules.augment_dataset

## Usage Examples

### Basic Graph Augmentation

```python
from augchem.modules.graph.graphs_modules import augment_dataset
import torch
from torch_geometric.data import Data

# Example: Create sample molecular graphs
graphs = [
    Data(x=torch.randn(10, 5), edge_index=torch.randint(0, 10, (2, 20))),
    Data(x=torch.randn(8, 5), edge_index=torch.randint(0, 8, (2, 16)))
]

# Apply multiple augmentation techniques
augmented_graphs = augment_dataset(
    graphs=graphs,
    augmentation_methods=['edge_drop', 'node_drop', 'feature_mask', 'edge_perturb'],
    edge_drop_rate=0.1,
    node_drop_rate=0.1, 
    feature_mask_rate=0.15,
    edge_add_rate=0.05,
    edge_remove_rate=0.05,
    augment_percentage=0.3,
    seed=42
)

print(f"Original: {len(graphs)} graphs")
print(f"Augmented: {len(augmented_graphs)} graphs")
```

### Individual Augmentation Techniques

```python
from augchem.modules.graph.graphs_modules import (
    edge_dropping, node_dropping, feature_masking, edge_perturbation
)

# Apply individual techniques
graph = your_molecular_graph

# Edge dropping - removes bidirectional connections
graph_edge_drop = edge_dropping(graph, drop_rate=0.1)

# Node dropping - removes nodes and their connections
graph_node_drop = node_dropping(graph, drop_rate=0.1)

# Feature masking - masks node features with -inf
graph_feature_mask = feature_masking(graph, mask_rate=0.15)

# Edge perturbation - adds and removes edges
graph_perturbed = edge_perturbation(
    graph, 
    add_rate=0.05, 
    remove_rate=0.05
)
```

### Working with PyTorch Geometric DataLoaders

```python
from torch_geometric.loader import DataLoader

# Create DataLoader with augmented graphs
dataloader = DataLoader(
    augmented_graphs,
    batch_size=32,
    shuffle=True
)

# Process batches
for batch in dataloader:
    print(f"Batch size: {batch.num_graphs}")
    print(f"Total nodes: {batch.x.size(0)}")
    print(f"Total edges: {batch.edge_index.size(1)}")
    break
```

## Technical Notes

### Graph Integrity
- All augmentation functions preserve graph structure validity
- Node indices are properly remapped after node dropping
- Edge attributes are handled consistently across operations

### Bidirectional Edges
- Edge dropping and perturbation work with complete bidirectional edges
- This ensures molecular graph connectivity is maintained properly
- Single-direction edge operations would break chemical bond representation

### Feature Masking
- Uses `-inf` as mask value for compatibility with attention mechanisms
- Masked features can be easily identified and handled in downstream models
- Preserves tensor shapes for batch processing

### Reproducibility
- All augmentation functions support random seed control
- Deterministic results for the same input parameters and seed
- Essential for experimental reproducibility in research

### Memory Efficiency
- All functions create cloned graphs to preserve originals
- Efficient tensor operations using PyTorch primitives
- Batch processing optimized for GPU acceleration
