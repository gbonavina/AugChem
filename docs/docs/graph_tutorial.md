# Graph Augmentation Tutorial

This tutorial demonstrates how to use AugChem's graph augmentation capabilities for molecular data enhancement.

## Prerequisites

```bash
pip install augchem torch torch-geometric rdkit
```

## Basic Setup

```python
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.transforms import NormalizeFeatures
from augchem.modules.graph.graphs_modules import (
    augment_dataset, edge_dropping, node_dropping, 
    feature_masking, edge_perturbation
)
```

## Tutorial 1: Loading Molecular Data

### Creating Sample Molecular Graphs

```python
# Create sample molecular graphs (representing small molecules)
def create_sample_graph():
    """Create a simple molecular graph (e.g., ethanol: C-C-O)"""
    # Node features: [atomic_number, degree, formal_charge]
    x = torch.tensor([[6, 1, 0], [6, 2, 0], [8, 1, 0]], dtype=torch.float)  # C, C, O
    # Edge connections: C-C, C-O bonds (bidirectional)
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    return Data(x=x, edge_index=edge_index, num_nodes=3)

# Create a small dataset of graphs
graphs = [create_sample_graph() for _ in range(10)]
print(f"Created {len(graphs)} sample molecular graphs")
```

### Understanding Graph Structure

```python
# Examine a single molecular graph
sample_graph = graphs[0]

print(f"Number of atoms (nodes): {sample_graph.num_nodes}")
print(f"Number of bonds (edges): {sample_graph.edge_index.size(1)}")
print(f"Node features shape: {sample_graph.x.shape}")

# Node features (per atom):
# [atomic_num, degree, formal_charge, hybridization, aromatic, 
#  radical_electrons, total_hydrogens, in_ring, mass]

# Edge features (per bond):
# [bond_type, aromatic, in_ring, conjugated]
```

### Loading from SDF Files (Optional)

```python
# If you have SDF files, you can use the TorchGeometricSDFLoader
# from your_loader import TorchGeometricSDFLoader

# loader = TorchGeometricSDFLoader(
#     sdf_path="path/to/your/molecules.sdf",
#     transform=NormalizeFeatures()
# )

# molecules = loader.load_sdf(max_molecules=1000)
# graphs = loader.convert_to_torch_geometric_graphs(add_self_loops=False)
```

## Tutorial 2: Individual Augmentation Techniques

### Edge Dropping

```python
# Remove 10% of molecular bonds
augmented_graph = edge_dropping(sample_graph, drop_rate=0.1)

print(f"Original bonds: {sample_graph.edge_index.size(1)}")
print(f"After edge dropping: {augmented_graph.edge_index.size(1)}")
print(f"Bonds removed: {sample_graph.edge_index.size(1) - augmented_graph.edge_index.size(1)}")
```

### Node Dropping

```python
# Remove 5% of atoms
augmented_graph = node_dropping(sample_graph, drop_rate=0.05)

print(f"Original atoms: {sample_graph.num_nodes}")
print(f"After node dropping: {augmented_graph.num_nodes}")
print(f"Atoms removed: {sample_graph.num_nodes - augmented_graph.num_nodes}")
```

### Feature Masking

```python
# Mask 15% of atomic features
augmented_graph = feature_masking(sample_graph, mask_rate=0.15)

# Check for masked features (they become -inf)
masked_features = (augmented_graph.x == float('-inf')).sum().item()
total_features = augmented_graph.x.numel()
mask_percentage = (masked_features / total_features) * 100

print(f"Masked features: {masked_features}/{total_features} ({mask_percentage:.1f}%)")
```

### Edge Perturbation

```python
# Add 3% new bonds and remove 3% existing bonds
augmented_graph = edge_perturbation(
    sample_graph, 
    add_rate=0.03, 
    remove_rate=0.03
)

print(f"Original bonds: {sample_graph.edge_index.size(1)}")
print(f"After perturbation: {augmented_graph.edge_index.size(1)}")
```

## Tutorial 3: Comprehensive Dataset Augmentation

### Basic Augmentation

```python
# Apply multiple techniques to create an augmented dataset
augmented_dataset = augment_dataset(
    graphs=graphs,
    augmentation_methods=['edge_drop', 'node_drop', 'feature_mask'],
    edge_drop_rate=0.1,
    node_drop_rate=0.05,
    feature_mask_rate=0.15,
    augment_percentage=0.25,  # 25% more data
    seed=42  # For reproducibility
)

print(f"Original dataset: {len(graphs)} graphs")
print(f"Augmented dataset: {len(augmented_dataset)} graphs")
print(f"New graphs added: {len(augmented_dataset) - len(graphs)}")
```

### Advanced Augmentation

```python
# Use all available techniques with custom parameters
augmented_dataset = augment_dataset(
    graphs=graphs,
    augmentation_methods=['edge_drop', 'node_drop', 'feature_mask', 'edge_perturb'],
    edge_drop_rate=0.12,
    node_drop_rate=0.08,
    feature_mask_rate=0.20,
    edge_add_rate=0.04,
    edge_remove_rate=0.06,
    augment_percentage=0.40,  # 40% more data
    seed=42
)

# Check augmentation metadata
for i, graph in enumerate(augmented_dataset[-10:]):  # Last 10 graphs
    if hasattr(graph, 'augmentation_method'):
        print(f"Graph {len(graphs) + i}: {graph.augmentation_method}")
        print(f"  Parent graph: {graph.parent_idx}")
```

### Graph Augmentation Methods Summary

| Method | Description | Parameters | Chemical Interpretation |
|--------|-------------|------------|------------------------|
| **edge_drop** | Remove molecular bonds | `drop_rate` | Bond breaking simulation |
| **node_drop** | Remove atoms | `drop_rate` | Atomic deletion |
| **feature_mask** | Mask atom properties | `mask_rate` | Property uncertainty |
| **edge_perturb** | Add/remove bonds | `add_rate`, `remove_rate` | Chemical reaction simulation |

## Tutorial 4: Batch Processing and Training

### Creating DataLoaders

```python
from sklearn.model_selection import train_test_split

# Split augmented dataset
train_graphs, test_graphs = train_test_split(
    augmented_dataset, 
    test_size=0.2, 
    random_state=42
)

# Create DataLoaders
train_loader = DataLoader(train_graphs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_graphs, batch_size=32, shuffle=False)

print(f"Training graphs: {len(train_graphs)}")
print(f"Testing graphs: {len(test_graphs)}")
```

### Analyzing Batches

```python
# Examine a training batch
for batch in train_loader:
    print(f"Batch contains {batch.num_graphs} graphs")
    print(f"Total nodes in batch: {batch.x.size(0)}")
    print(f"Total edges in batch: {batch.edge_index.size(1)}")
    print(f"Node features shape: {batch.x.shape}")
    
    # The batch tensor maps nodes to their respective graphs
    print(f"Batch assignment shape: {batch.batch.shape}")
    break
```

## Tutorial 5: Quality Control and Validation

### Self-Loop Detection

```python
# Check for self-loops before augmentation
def check_self_loops(graphs):
    total_self_loops = 0
    for graph in graphs:
        if graph.edge_index.size(1) > 0:
            self_loops = (graph.edge_index[0] == graph.edge_index[1]).sum().item()
            total_self_loops += self_loops
    return total_self_loops

original_self_loops = check_self_loops(graphs)
augmented_self_loops = check_self_loops(augmented_dataset)

print(f"Self-loops in original dataset: {original_self_loops}")
print(f"Self-loops in augmented dataset: {augmented_self_loops}")
```

### Graph Statistics

```python
# Analyze dataset statistics
def analyze_dataset(graphs, name):
    num_nodes = [g.num_nodes for g in graphs]
    num_edges = [g.edge_index.size(1) for g in graphs]
    
    print(f"\n{name} Dataset Statistics:")
    print(f"  Total graphs: {len(graphs)}")
    print(f"  Nodes per graph: {sum(num_nodes)/len(num_nodes):.1f} ± {torch.tensor(num_nodes).float().std():.1f}")
    print(f"  Edges per graph: {sum(num_edges)/len(num_edges):.1f} ± {torch.tensor(num_edges).float().std():.1f}")

analyze_dataset(graphs, "Original")
analyze_dataset(augmented_dataset, "Augmented")
```

### Graph Validation

```python
def validate_graphs(graph_list):
    """Validate molecular graphs"""
    stats = {
        'total': len(graph_list),
        'avg_nodes': sum(g.num_nodes for g in graph_list) / len(graph_list),
        'avg_edges': sum(g.edge_index.size(1) for g in graph_list) / len(graph_list),
        'isolated_nodes': 0,
        'empty_graphs': 0
    }
    
    for graph in graph_list:
        # Check for isolated nodes (nodes with no edges)
        if graph.edge_index.size(1) == 0 and graph.num_nodes > 0:
            stats['isolated_nodes'] += 1
        
        # Check for empty graphs
        if graph.num_nodes == 0:
            stats['empty_graphs'] += 1
    
    return stats

# Validate augmented graphs
graph_stats = validate_graphs(augmented_dataset)
print("Graph Statistics:")
for key, value in graph_stats.items():
    if isinstance(value, float):
        print(f"  {key}: {value:.2f}")
    else:
        print(f"  {key}: {value}")
```

## Tutorial 6: Visualization and Analysis

### Augmentation Impact Analysis

```python
import matplotlib.pyplot as plt

# Compare distributions
original_nodes = [g.num_nodes for g in graphs]
augmented_nodes = [g.num_nodes for g in augmented_dataset]

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.hist(original_nodes, bins=30, alpha=0.7, label='Original', color='blue')
plt.hist(augmented_nodes, bins=30, alpha=0.7, label='Augmented', color='red')
plt.xlabel('Number of Nodes')
plt.ylabel('Frequency')
plt.title('Node Count Distribution')
plt.legend()

plt.subplot(1, 2, 2)
original_edges = [g.edge_index.size(1) for g in graphs]
augmented_edges = [g.edge_index.size(1) for g in augmented_dataset]

plt.hist(original_edges, bins=30, alpha=0.7, label='Original', color='blue')
plt.hist(augmented_edges, bins=30, alpha=0.7, label='Augmented', color='red')
plt.xlabel('Number of Edges')
plt.ylabel('Frequency')
plt.title('Edge Count Distribution')
plt.legend()

plt.tight_layout()
plt.show()
```

### Advanced Graph Analysis

```python
def comprehensive_graph_analysis(graphs, name):
    """Comprehensive analysis of graph dataset"""
    
    # Basic statistics
    num_nodes = [g.num_nodes for g in graphs]
    num_edges = [g.edge_index.size(1) for g in graphs]
    
    # Calculate degree statistics
    degrees = []
    for graph in graphs:
        if graph.edge_index.size(1) > 0:
            from torch_geometric.utils import degree
            deg = degree(graph.edge_index[0], num_nodes=graph.num_nodes)
            degrees.extend(deg.tolist())
    
    stats = {
        'num_graphs': len(graphs),
        'avg_nodes': sum(num_nodes) / len(num_nodes),
        'avg_edges': sum(num_edges) / len(num_edges),
        'avg_degree': sum(degrees) / len(degrees) if degrees else 0,
        'max_nodes': max(num_nodes) if num_nodes else 0,
        'max_edges': max(num_edges) if num_edges else 0
    }
    
    print(f"\n{name} Analysis:")
    for key, value in stats.items():
        print(f"  {key}: {value:.2f}")
    
    return stats

# Comprehensive analysis
original_stats = comprehensive_graph_analysis(graphs, "Original")
augmented_stats = comprehensive_graph_analysis(augmented_dataset, "Augmented")
```

## Tutorial 7: Integration with Graph Neural Networks

### Simple GNN Example

```python
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool

class MolecularGNN(nn.Module):
    def __init__(self, num_features, hidden_dim=64, num_classes=1):
        super().__init__()
        self.conv1 = GCNConv(num_features, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.classifier = nn.Linear(hidden_dim, num_classes)
        
    def forward(self, x, edge_index, batch):
        # Graph convolutions
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        
        # Global pooling
        x = global_mean_pool(x, batch)
        
        # Classification
        return self.classifier(x)

# Initialize model
model = MolecularGNN(num_features=graphs[0].x.size(1))

# Training loop example
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

model.train()
for epoch in range(5):
    total_loss = 0
    for batch in train_loader:
        optimizer.zero_grad()
        
        # Forward pass
        out = model(batch.x, batch.edge_index, batch.batch)
        
        # Dummy target for demonstration
        target = torch.randn(batch.num_graphs, 1)
        
        # Loss and backprop
        loss = criterion(out, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}, Loss: {total_loss/len(train_loader):.4f}")
```

### Advanced GNN with Augmented Data

```python
# More sophisticated training with augmented data
def train_with_augmentation():
    """Training loop that leverages augmented data"""
    
    # Create augmented dataset with labels
    augmented_graphs_with_targets = []
    for graph in augmented_dataset:
        graph.y = torch.randn(1)  # Random target for demo
        augmented_graphs_with_targets.append(graph)
    
    # Split and create loaders
    train_graphs, val_graphs = train_test_split(
        augmented_graphs_with_targets, test_size=0.2, random_state=42
    )
    
    train_loader = DataLoader(train_graphs, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=16, shuffle=False)
    
    model = MolecularGNN(num_features=graphs[0].x.size(1))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()
    
    # Training with validation
    for epoch in range(10):
        # Train
        model.train()
        train_loss = 0
        for batch in train_loader:
            optimizer.zero_grad()
            out = model(batch.x, batch.edge_index, batch.batch)
            loss = criterion(out, batch.y.view(-1, 1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        
        # Validate
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                out = model(batch.x, batch.edge_index, batch.batch)
                loss = criterion(out, batch.y.view(-1, 1))
                val_loss += loss.item()
        
        print(f"Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Loss: {val_loss/len(val_loader):.4f}")

# Run advanced training
# train_with_augmentation()
```

## Best Practices for Graph Augmentation

### 1. Data Preparation
- Always check for self-loops before augmentation
- Use transforms without `AddSelfLoops` for augmentation workflows
- Validate graph integrity after augmentation

### 2. Augmentation Strategy
- Start with conservative augmentation rates (5-15%)
- Monitor the impact on model performance
- Use different techniques for different aspects of robustness

### 3. Reproducibility
- Always set random seeds for experiments
- Document augmentation parameters
- Save augmented datasets for reuse

### 4. Performance Optimization
- Use appropriate batch sizes for your hardware
- Profile memory usage for large datasets
- Consider distributed processing for very large datasets

### 5. Validation
- Compare augmented vs original performance
- Use cross-validation with consistent augmentation
- Monitor for data leakage between train/test sets

## Common Issues and Solutions

### Self-loops Causing Errors
```python
# Remove self-loops before augmentation
from torch_geometric.utils import remove_self_loops

def clean_graphs(graphs):
    cleaned = []
    for graph in graphs:
        edge_index, edge_attr = remove_self_loops(graph.edge_index, graph.edge_attr)
        cleaned_graph = Data(
            x=graph.x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            num_nodes=graph.num_nodes
        )
        cleaned.append(cleaned_graph)
    return cleaned

# Apply cleaning
clean_graphs_list = clean_graphs(graphs)
```

### Memory Issues
```python
# Process in smaller batches
def memory_efficient_augmentation(graphs, batch_size=100):
    all_augmented = []
    
    for i in range(0, len(graphs), batch_size):
        batch_graphs = graphs[i:i+batch_size]
        
        batch_augmented = augment_dataset(
            graphs=batch_graphs,
            augmentation_methods=['edge_drop'],
            augment_percentage=0.2
        )
        
        all_augmented.extend(batch_augmented)
        print(f"Processed batch {i//batch_size + 1}")
    
    return all_augmented

# Use for large datasets
# large_augmented = memory_efficient_augmentation(large_graph_list)
```

### Edge Attribute Mismatches
```python
# Validate edge attributes
def validate_edge_attributes(graphs):
    for i, graph in enumerate(graphs):
        if hasattr(graph, 'edge_attr') and graph.edge_attr is not None:
            if graph.edge_attr.size(0) != graph.edge_index.size(1):
                print(f"Graph {i}: Edge attribute mismatch!")
                print(f"  Edge index: {graph.edge_index.size(1)} edges")
                print(f"  Edge attr: {graph.edge_attr.size(0)} attributes")

# Check before augmentation
validate_edge_attributes(graphs)
```

## Saving and Loading Augmented Graphs

```python
# Save augmented dataset
torch.save(augmented_dataset, "augmented_molecular_graphs.pt")
print(f"Saved {len(augmented_dataset)} graphs")

# Load augmented dataset
loaded_graphs = torch.load("augmented_molecular_graphs.pt")
print(f"Loaded {len(loaded_graphs)} graphs")

# Save with metadata
augmentation_info = {
    'graphs': augmented_dataset,
    'parameters': {
        'edge_drop_rate': 0.1,
        'node_drop_rate': 0.05,
        'augment_percentage': 0.25,
        'seed': 42
    },
    'original_count': len(graphs),
    'augmented_count': len(augmented_dataset)
}

torch.save(augmentation_info, "augmented_graphs_with_metadata.pt")
```

## Next Steps

After completing this graph tutorial:

- ✅ **Master all graph augmentation techniques**
- ✅ **Implement quality control and validation**
- ✅ **Integrate with PyTorch Geometric workflows**
- ✅ **Build and train Graph Neural Networks**
- ✅ **Handle large molecular datasets efficiently**

### See Also

- [SMILES Augmentation Tutorial](smiles_tutorial.md) - Learn string-based augmentation
- [Graph Methods API](reference/graphs_methods.md) - Complete method documentation
- [Examples](examples.md) - Real-world applications
