# Data Loaders

This section covers the data loading utilities available in Augchem. These tools help you load and process molecular data from various formats for augmentation and analysis.

## Standard Loader

This class is a helper in case you don't have your own QM9 dataset loader. In case you have your own dataset loader, you can ignore these functions.

It's important to note that these data augmentation methods aren't exclusive to the QM9 dataset, and can be used with any SMILES, graph or InChI data.

::: augchem.QM9Loader
    options:
      show_source: true

## PyTorch Geometric SDF Loader

The `TorchGeometricSDFLoader` is a specialized loader for working with SDF (Structure-Data File) format using PyTorch Geometric. This loader is optimized for molecular graph processing and augmentation workflows.

### Key Features

- **Pure PyTorch Geometric**: Uses only PyTorch Geometric primitives for optimal performance
- **SDF Support**: Direct loading from SDF files containing molecular structures
- **Graph Conversion**: Converts RDKit molecules to `torch_geometric.data.Data` objects
- **Batch Processing**: Integrated DataLoader creation for efficient training
- **Analytics**: Comprehensive graph statistics and visualization tools
- **Self-loop Management**: Automatic detection and removal for clean augmentation

### Example Usage

```python
# Initialize the loader
loader = TorchGeometricSDFLoader(
    sdf_path="path/to/molecules.sdf",
    transform=NormalizeFeatures()  # Optional transforms
)

# Load molecules from SDF
molecules = loader.load_sdf(max_molecules=1000)

# Convert to PyTorch Geometric graphs
graphs = loader.convert_to_torch_geometric_graphs(
    add_self_loops=False  # Important for augmentation techniques
)

# Create DataLoader for batch processing
dataloader = loader.create_torch_geometric_dataloader(
    batch_size=32, 
    shuffle=True
)

# Get comprehensive statistics
stats = loader.get_torch_geometric_statistics()
print(f"Total graphs: {stats['total_graphs']}")
print(f"Average nodes per graph: {stats['nodes_per_graph']['mean']:.2f}")

# Visualize a molecular graph
loader.visualize_with_torch_geometric(idx=0)

# Save processed graphs
loader.save_torch_geometric_graphs("processed_graphs.pt")
```

### Graph Features

#### Node Features (9 dimensions)
1. Atomic number
2. Degree (number of bonds)
3. Formal charge
4. Hybridization type
5. Aromaticity (boolean)
6. Number of radical electrons
7. Total number of hydrogens
8. Ring membership (boolean)
9. Atomic mass

#### Edge Features (4 dimensions)
1. Bond type (single, double, triple, aromatic)
2. Aromaticity (boolean)
3. Ring membership (boolean)
4. Conjugation (boolean)

### Augmentation-Ready Processing

The loader is specifically designed to work seamlessly with the graph augmentation techniques:

```python
# Load and prepare graphs for augmentation
loader = TorchGeometricSDFLoader(sdf_path, transform=None)
molecules = loader.load_sdf(max_molecules=500)

# Convert without self-loops (optimal for edge dropping)
graphs = loader.convert_to_torch_geometric_graphs(add_self_loops=False)

# Apply augmentation techniques
from augchem.modules.graph.graphs_modules import augment_dataset

augmented_graphs = augment_dataset(
    graphs=graphs,
    augmentation_methods=['edge_drop', 'node_drop', 'feature_mask'],
    augment_percentage=0.3
)
```

### Advanced Analytics

```python
# Check for self-loops before augmentation
from your_utils import check_self_loops_in_graphs, remove_self_loops_from_graphs

# Analyze self-loops
self_loop_stats = check_self_loops_in_graphs(graphs)
print(f"Graphs with self-loops: {self_loop_stats['graphs_with_self_loops']}")

# Clean graphs if needed
if self_loop_stats['total_self_loops'] > 0:
    cleaned_graphs = remove_self_loops_from_graphs(graphs)
    loader.graphs = cleaned_graphs  # Update loader with clean graphs

# Comprehensive visualization
plot_torch_geometric_statistics(cleaned_graphs)
```

### Performance Tips

1. **Memory Management**: Load molecules in batches for large datasets
2. **Self-loop Removal**: Always check and remove self-loops before augmentation
3. **Transform Selection**: Use transforms without `AddSelfLoops` for augmentation workflows
4. **Batch Size**: Optimize batch size based on your GPU memory
5. **Reproducibility**: Set random seeds for consistent results

### Integration with Machine Learning

```python
# Create train/validation splits
from sklearn.model_selection import train_test_split

train_graphs, val_graphs = train_test_split(
    augmented_graphs, 
    test_size=0.2, 
    random_state=42
)

# Create optimized DataLoaders
train_loader = DataLoader(train_graphs, batch_size=64, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=64, shuffle=False)

# Ready for GNN training!
for batch in train_loader:
    # batch.x: node features
    # batch.edge_index: edge connections
    # batch.edge_attr: edge features
    # batch.batch: graph assignment for nodes
    pass
```